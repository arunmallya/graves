--[[
An implementation of the methods from http://deeplearning.net/tutorial/lstm.html
The architecture is the same except that the Mean Pooling layer is replaced
by a Sum layer. This makes for faster convergence 
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

local SentimentMinibatchLoader = require 'util.SentimentMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'models.LSTM'
local RNN = require 'models.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')

-- data
cmd:option('-data_dir','data/imdb','data directory. Should contain the files {train, test}.txt with input data')

-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')

-- optimization
cmd:option('-learning_rate', 0.0002, 'learning rate')
cmd:option('-learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_seq_length', 100, 'maximum number of timesteps to unroll for')
cmd:option('-vocab_size', 10000, 'number of words in input')
cmd:option('-batch_size', 10, 'number of sequences to train on in parallel')
cmd:option('-max_epochs', 50, 'number of full passes through the training data')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-train_frac', 0.95, 'fraction of data that goes into train set')
cmd:option('-val_frac', 0.05, 'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')

-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 120, 'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile', 'lstm', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing', 0, 'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')

-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-opencl', 0, 'use OpenCL (instead of CUDA)')
cmd:text()


--------------- INITIALIZE ---------------

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
if not torch.isTensor(opt.rnn_size) then 
    opt.rnn_size = torch.Tensor(opt.num_layers):fill(opt.rnn_size)
end
assert(opt.rnn_size:size(1) == opt.num_layers, 'invalid rnn_size: need one scalar or a tensor of same length as num_layers')
-- train / val split for data, in fractions
local split_sizes = {opt.train_frac, opt.val_frac} 

-- initialize cuda for training
opt = init_cuda(opt)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- create the data loader class
local loader = SentimentMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.max_seq_length, opt.vocab_size, split_sizes)
vocab_size   = opt.vocab_size  -- the number of distinct characters
print('vocab size: ' .. vocab_size)
-- create the input vector generator
input_gen = OneHot(vocab_size)


--------------- DEFINE THE MODEL ---------------

function define_model(opt)

    opt.do_random_init = true
    local protos = {}

    -- define the model: prototypes for one timestep, then clone them in time
    if string.len(opt.init_from) > 0 then
        print('loading an LSTM from checkpoint ' .. opt.init_from)
        local checkpoint = torch.load(opt.init_from)
        protos = checkpoint.protos
        -- make sure the vocabs are the same
        local vocab_compatible = true
        for c,i in pairs(checkpoint.vocab) do 
            if not vocab[c] == i then 
                vocab_compatible = false
            end
        end
        assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
        -- overwrite model settings based on checkpoint to ensure compatibility
        print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
        opt.rnn_size = checkpoint.opt.rnn_size
        opt.num_layers = checkpoint.opt.num_layers
        opt.do_random_init = false
    else
        -- create the rnn
        rnn_opts = {}
        print('\nCreating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
        if opt.model == 'lstm' then
            inputs, outputs = LSTM.create(vocab_size, opt.num_layers, opt.rnn_size, rnn_opts)
            protos.rnn = nn.gModule(inputs, outputs)
        elseif opt.model == 'rnn' then
            inputs, outputs = RNN.create(vocab_size, opt.num_layers, opt.rnn_size, rnn_opts)
            protos.rnn = nn.gModule(inputs, outputs)
        end
        
        -- create the prediction layer
        local m = nn.Sequential()
        m:add(nn.CAddTable())
        m:add(nn.Dropout(0.5))
        m:add(nn.Linear(opt.rnn_size[opt.num_layers], 2))
        m:add(nn.LogSoftMax())
        protos.mean_pred = m

        -- create the criterion
        protos.criterion = nn.ClassNLLCriterion()
    end

    return protos, opt
end

--------------- INITIALIZE THE MODEL (state & params) ---------------

function init_model(opt, protos, input_gen)
    -- the initial state of the cell/hidden states
    init_state = {}
    for L = 1, opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.rnn_size[L])
        if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(init_state, h_init:clone())
        if opt.model == 'lstm' then
            table.insert(init_state, h_init:clone())
        end
    end

    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 and opt.opencl == 0 then
        for k,v in pairs(protos) do v:cuda() end
        input_gen:cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then
        for k,v in pairs(protos) do v:cl() end
        input_gen:cl()
    end

    -- put the above things into one flattened parameters tensor
    local params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.mean_pred)

    -- initialization of rnn network params
    if opt.do_random_init then
        params:uniform(-0.08, 0.08) -- small uniform numbers
    end
    -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
    if opt.model == 'lstm' then
        for layer_idx = 1, opt.num_layers do
            for _,node in ipairs(protos.rnn.forwardnodes) do
                if node.data.annotations.name == 'i2h_' .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size[layer_idx]+1, 2*opt.rnn_size[layer_idx]}}]:fill(1.0)
                end
            end
        end
    end

    return protos, input_gen, init_state, params, grad_params
end

--------------- MODEL DRIVER ---------------

 -- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local num_correct = 0
    local indices
    local rnn_state = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(opt, x, y)

        local current_seq_len = x:size(1)
        local hidden_outputs = {}
        protos.mean_pred:evaluate()
        for t=1,current_seq_len do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            input_vector = input_gen:forward(x[t])
            local lst = clones.rnn[t]:forward({input_vector, unpack(rnn_state[t-1])})
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            hidden_outputs[t] = lst[#lst]
            -- scale inputs ?
            --hidden_outputs[t]:mul(1 / current_seq_len) 
        end

        local predictions = protos.mean_pred:forward(hidden_outputs)
        _, indices = torch.max(predictions, 2)

        num_correct = num_correct + torch.sum(torch.eq(indices, y))

        print(i .. '/' .. n .. '...')
    end

    local accuracy = num_correct / (n * opt.batch_size)
    return accuracy
end

-- do fwd/bwd and return loss, grad_params
function modelEval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = prepro(opt, x, y)
    local current_seq_len = x:size(1)

    ------------------- forward pass - RNN -------------------
    local rnn_state = {[0] = init_state_global}
    local hidden_outputs = {}
    local loss = 0
    local input_vector

    -- prepare input data
    local inputs = {}
    for t=1,current_seq_len do
        inputs[t] = input_gen:forward(x[t]):clone()
    end

    protos.mean_pred:training()
    for t=1,current_seq_len do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)

        local lst = clones.rnn[t]:forward({inputs[t], unpack(rnn_state[t-1])})

        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

        -- last element is the hidden output of rnn    
        hidden_outputs[t] = lst[#lst]
    end
    

    ------------------- forward pass - mean_pred -------------------
    local predictions = protos.mean_pred:forward(hidden_outputs)
    loss = loss + protos.criterion:forward(predictions, y)

    ------------------ backward pass - mean_pred -------------------    
    local doutput  = protos.criterion:backward(predictions, y)
    local dhidouts = protos.mean_pred:backward(hidden_outputs, doutput)

    ------------------ backward pass - RNN -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[current_seq_len] = clone_list(init_state, true)} -- true also zeros the clones
    for t=current_seq_len,1,-1 do
        -- backprop through loss, and softmax/linear
        drnn_state[t][#drnn_state[t]]:add(dhidouts[t])

        local dlst = clones.rnn[t]:backward({inputs[t], unpack(rnn_state[t-1])}, drnn_state[t])
        
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- in our case, there is no continuation among batches
    -- thus the state should be reset to 0 instead of carrying forward 
    -- this is in contrast to the char-rnn case, where there is
    -- continuity over batches
    -- init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(current_seq_len) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

--------------------------------------------

-- define the model
protos, opt = define_model(opt)

print(protos)

-- initialize the model
protos, input_gen, init_state, params, grad_params = init_model(opt, protos, input_gen)
init_state_global = clone_list(init_state)
print('number of parameters in the model: ' .. params:nElement())

-- make a bunch of clones of rnn after flattening, as that reallocates memory
clones = {}
print('cloning rnn')
clones['rnn'] = model_utils.clone_many_times(protos.rnn, opt.max_seq_length, not protos.rnn.parameters)

-- start optimization here
train_losses = {}
val_accuracies = {}
test_accuracies = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    -- perform rmsprop 
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(modelEval, params, optim_state)
    --local _, loss = optim.adadelta(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.           learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_accuracy = eval_split(2) -- 2 = validation
        local test_accuracy = eval_split(3) -- 3 = test
        val_accuracies[i] = val_accuracy
        test_accuracies[i] = test_accuracy

        local savefile = string.format('%s/lm_%s_epoch%.2f_val%.4f_test%.4f.t7', opt.checkpoint_dir, opt.  savefile, epoch, val_accuracy, test_accuracy)
        print('saving checkpoint to ' .. savefile)
                local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.vocab_size = vocab_size
        checkpoint.train_losses = train_losses
        checkpoint.val_accuracy = val_accuracy
        checkpoint.val_accuracies = val_accuracies
        checkpoint.test_accuracy = test_accuracy
        checkpoint.test_accuracies = test_accuracies
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch   = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing      issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-    bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end