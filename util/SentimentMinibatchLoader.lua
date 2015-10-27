local SentimentMinibatchLoader = {}
SentimentMinibatchLoader.__index = SentimentMinibatchLoader

function SentimentMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.1}

    local self = {}
    setmetatable(self, SentimentMinibatchLoader)

    local train_file  = path.join(data_dir, 'train.txt')
    local test_file   = path.join(data_dir, 'test.txt')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('data.t7 does not exist. Running preprocessing...')
        run_prepro = true
    end

    if run_prepro then
        -- construct a tensor with all the data
        print('one-time setup: preprocessing input text files ' .. train_file .. ', ' .. test_file .. '...')
        SentimentMinibatchLoader.text_to_tensor(train_file, test_file, tensor_file)
    end

    print('loading data files...')
    local data  = torch.load(tensor_file)
    all_train_data  = data.train_data
    all_train_label = data.train_label
    all_test_data   = data.test_data
    all_test_label  = data.test_label

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    -- drop all examples with length greater than seq_length
    -- all maintain list of size of each example for efficient batching
    train_data  = {}
    train_label = {}
    train_lens  = {}
    index = 1
    for i = 1, #all_train_data do
        if all_train_data[i]:size(1) <= self.seq_length then
            train_data[index]  = all_train_data[i]
            train_label[index] = all_train_label[i]
            train_lens[index]  = all_train_data[i]:size(1)
            index = index + 1
        end
    end
    test_data  = {}
    test_label = {}
    test_lens  = {}
    index = 1
    for i = 1, #all_test_data do
        if all_test_data[i]:size(1) <= self.seq_length then
            test_data[index]  = all_test_data[i]
            test_label[index] = all_test_label[i]
            test_lens[index]  = all_test_data[i]:size(1)
            index = index + 1
        end
    end
    print('#train = ' .. #train_data)
    print('#test = ' .. #test_data)
    self.train_data  = train_data
    self.train_label = train_label
    self.test_data   = test_data
    self.test_label  = test_label

    -- get sorted example size ordering
    -- will use this for efficient batching and padding each batch
    train_lens = torch.Tensor(train_lens)
    test_lens = torch.Tensor(test_lens)
    train_lens, train_ind = torch.sort(train_lens)
    test_lens, test_ind  = torch.sort(test_lens)
    self.train_ind  = train_ind
    self.test_ind   = test_ind
    self.train_lens = train_lens
    self.test_lens  = test_lens

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')

    -- divide data to train/val and allocate rest to test
    local ntrainbatches = math.floor(#train_data / self.batch_size)
    self.ntest = math.floor(#test_data / self.batch_size)
    self.ntrain = math.floor(ntrainbatches * split_fractions[1])
    self.nval = math.floor(ntrainbatches * split_fractions[2])
    
    -- create a permutation of batch ordering
    -- we have sorted the length of samples so as to create batches 
    -- in which all examples have similar lengths (for efficiency)
    -- at the same time, we want consecutive batches to have random
    -- max_len values. Thus we create a random permutation of batch
    -- indices to follow
    self.perm_order = torch.randperm(self.ntrain + self.nval)

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function SentimentMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function SentimentMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 1 then ix = self.perm_order[ix] end
    if split_index == 2 then ix = self.perm_order[ix + self.ntrain] end -- offset by train set size
    
    local start_ind = (ix-1) * self.batch_size + 1
    local end_ind   = start_ind + self.batch_size - 1
    local data, label

    if split_index == 3 then
        -- return from test data
        local indices = self.test_ind[{{start_ind, end_ind}}]       
        local lens = self.test_lens[{{start_ind, end_ind}}]       
        local max_len = torch.max(lens)
        data  = torch.Tensor(self.batch_size, max_len)
        label = torch.Tensor(self.batch_size)

        for i = 1, self.batch_size do
            local item = self.test_data[indices[i]]
            local pad = max_len - (#item)[1]
            if pad > 0 then
                item = torch.cat(item:float(), torch.FloatTensor(pad):zero())
            end
            data[{i, {}}] = item
            label[i] = self.test_label[indices[i]]
        end
    else 
        -- return from train data 
        local indices = self.train_ind[{{start_ind, end_ind}}]       
        local lens = self.train_lens[{{start_ind, end_ind}}]       
        local max_len = torch.max(lens)
        data  = torch.Tensor(self.batch_size, max_len)
        label = torch.Tensor(self.batch_size)

        for i = 1, self.batch_size do
            local item = self.train_data[indices[i]]
            local pad = max_len - (#item)[1]
            if pad > 0 then
                item = torch.cat(item:float(), torch.FloatTensor(pad):zero())
            end
            data[{i, {}}] = item
            label[i] = self.train_label[indices[i]]
        end
    end

    return data, label
end

-- *** STATIC method ***
function SentimentMinibatchLoader.text_to_tensor(train_textfile, test_textfile, out_tensorfile)
    local timer = torch.Timer()

    print('loading text files...')
    local cache_len = 10000
    local rawdata
    local tot_len = 0
    f_train = io.open(train_textfile, "r")
    f_test  = io.open(test_textfile, "r")

    local num_train = 0
    for _ in io.lines(train_textfile) do
      num_train = num_train + 1
    end
    local num_test = 0
    for _ in io.lines(test_textfile) do
      num_test = num_test + 1
    end

    -- construct a tensor with all the data
    print('putting data into tensor...')
    local train_data  = {}
    local train_label = torch.ByteTensor(num_train)
    local i = 0  
    for line in f_train:lines('*l') do  
      i = i + 1
      local l = line:split(',')
      sample = {}
      for key, val in ipairs(l) do
        sample[key] = val
      end
      sample = torch.ByteTensor(sample)
      train_label[i] = sample[1]
      train_data[i]  = sample[{{2,sample:size(1)}}]
    end
    f_train:close()  

    local test_data  = {}
    local test_label = torch.ByteTensor(num_test)
    local i = 0  
    for line in f_test:lines('*l') do  
      i = i + 1
      local l = line:split(',')
      sample = {}
      for key, val in ipairs(l) do
        sample[key] = val
      end
      sample = torch.ByteTensor(sample)
      test_label[i] = sample[1]
      test_data[i]  = sample[{{2,sample:size(1)}}]
    end
    f_test:close()

    data = {}
    data['train_data']  = train_data
    data['train_label'] = train_label
    data['test_data']   = test_data
    data['test_label']  = test_label

    -- save output preprocessed files
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

return SentimentMinibatchLoader

