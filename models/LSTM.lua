--[[ 
This implements two types of LSTMs: 
1) 'alex' -  http://arxiv.org/pdf/1308.0850v5.pdf
    layer 1 has access to prev_h[1] and x_t
    layer L(>1) has access to prev_h[L], next_h[L-1] and x_t
    layers also use prev_c[L] and next_c[L] for gate computation
    See Alex_LSTM.png in imgs/, from http://www.cs.toronto.edu/~graves/gen_seq_rnn.pdf
2) 'simplified' - adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
    layer 1 has access to prev_h[1] and x_t
    layer L(>1) has access to prev_h[L] and next_h[L-1]
    layers do not use c for gate computation
    
batch_size >= 1 is supported by this code
--]]

require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, num_L, num_h, rnn_type)

    -- check input sanity
    rnn_type = rnn_type or 'simplified'
    assert(rnn_type == 'simplified' or rnn_type == 'alex', 
        'invalid rnn type: '..rnn_type)
    assert(input_size > 0, 'invalid input_size: should be > 0')
    assert(num_L >= 1, 'invalid num_L: should be >= 1')
    if not torch.isTensor(num_h) then 
        num_h = torch.Tensor(num_L):fill(num_h)
    else
        assert(num_h:size(1) == num_L,
            [[invalid num_h: need one scalar or 
            a tensor of same length as num_L]])
    end

    -- build the required LSTM graph
    if rnn_type == 'simplified' then
        return LSTM.create_simplified(input_size, num_L, num_h)
    elseif rnn_type == 'alex' then
        return LSTM.create_alex(input_size, num_L, num_h)
    end

end

function LSTM.create_simplified(input_size, num_L, num_h)

    -- there are 1+n+n inputs: x_t, prev_h[1], ..., prev_h[n], prev_c[1], ..., prev_c[n]
    local inputs = {}
    table.insert(inputs, nn.Identity()():annotate{'x'})
    for L = 1, num_L do
        table.insert(inputs, nn.Identity()():annotate{'prev_c_'..L})
        table.insert(inputs, nn.Identity()():annotate{'prev_h_'..L})
    end

    -- build the computation graph of each layer
    local outputs = {}
    local x = inputs[1]
    local input, input_size_L, rnn_size
    for L = 1, num_L do
        rnn_size = num_h[L]

        -- inputs for this layer
        if L == 1 then
            input = x
            input_size_L = input_size
        else
            input = outputs[(L-1)*2] -- next_h[L-1]
            input_size_L = num_h[L-1]
        end

        -- c, h of the current layer, from previous timesteps
        local prev_c = inputs[L*2]
        local prev_h = inputs[L*2+1]

        -- matrix multiplication
        local i2h = nn.Linear(input_size_L, 4*rnn_size)(input):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, 4*rnn_size)(prev_h):annotate{name='h2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})
        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

        -- decode the gates
        local in_gate = nn.Sigmoid()(n1):annotate{name='i_'..L}
        local forget_gate = nn.Sigmoid()(n2):annotate{name='f_'..L}
        local out_gate = nn.Sigmoid()(n3):annotate{name='o_'..L}
        
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        
        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
        }):annotate{name='c_'..L}
        
        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate{name='h_'..L}
        
        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    -- create a module out of the computation graph
    return nn.gModule(inputs, outputs)

end

function LSTM.create_alex(input_size, num_L, num_h)

    -- there are 1+n+n inputs: x_t, prev_h[1], ..., prev_h[n], prev_c[1], ..., prev_c[n]
    local inputs = {}
    table.insert(inputs, nn.Identity()():annotate{'x'})
    for L = 1, num_L do
        table.insert(inputs, nn.Identity()():annotate{'prev_c_'..L})
        table.insert(inputs, nn.Identity()():annotate{'prev_h_'..L})
    end

    -- build the computation graph of each layer
    local outputs = {}
    local x = inputs[1]
    local input, input_size_L, rnn_size
    for L = 1, num_L do
        rnn_size = num_h[L]

        -- inputs for this layer
        if L == 1 then
            input = x
            input_size_L = input_size
        else
            input = nn.JoinTable(2)({x, outputs[(L-1)*2]}) -- [x, next_h[L-1]]
            input_size_L = num_h[L-1] + input_size
        end

        -- c, h of the current layer, from previous timesteps
        local prev_c = inputs[L*2]
        local prev_h = inputs[L*2+1]

        -- matrix multiplication
        local i2h = nn.Linear(input_size_L, 4*rnn_size)(input):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, 4*rnn_size)(prev_h):annotate{name='h2h_'..L}
        local c2h = nn.Linear(rnn_size, 2*rnn_size)(prev_c):annotate{name='c2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})
        local reshaped_input = nn.Reshape(4, rnn_size)(all_input_sums)
        local reshaped_c = nn.Reshape(2, rnn_size)(c2h)
        local i1, i2, i3, n4 = nn.SplitTable(2)(reshaped_input):split(4)
        local c1, c2 = nn.SplitTable(2)(reshaped_c):split(2)

        local n1 = nn.CAddTable()({i1, c1})
        local n2 = nn.CAddTable()({i2, c2})

        -- decode the input and forget gates
        local in_gate = nn.Sigmoid()(n1):annotate{name='i_'..L}
        local forget_gate = nn.Sigmoid()(n2):annotate{name='f_'..L}

        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)

        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
        }):annotate{name='c_'..L}

        -- decode the output gate
        local cnew2h = nn.Linear(rnn_size, rnn_size)(next_c):annotate{name='cnew2h_'..L}
        local n3 = nn.CAddTable()({i3, cnew2h})
        local out_gate = nn.Sigmoid()(n3):annotate{name='o_'..L}

        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate{name='h_'..L}

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    -- create a module out of the computation graph
    return nn.gModule(inputs, outputs)

end

function LSTM.test()

    batch_size = 1

    model = LSTM.create(1000, 1, 100)
    dummy_input  = torch.rand(batch_size, 1000)
    dummy_prev_c = torch.rand(batch_size, 100)
    dummy_prev_h = torch.rand(batch_size, 100)
    y = model:forward({dummy_input, dummy_prev_h, dummy_prev_h})
    print('Single Layer LSTM Model')
    print('batch_size: '..batch_size..', layers: 1')
    print(y)
    graph.dot(model.fg, 'Single Layer LSTM')


    simple_model = LSTM.create(1000, 2, 100)
    dummy_input  = torch.rand(batch_size, 1000)
    dummy_prev_c = torch.rand(batch_size, 100)
    dummy_prev_h = torch.rand(batch_size, 100)
    y = simple_model:forward({dummy_input, dummy_prev_c, dummy_prev_h, dummy_prev_c, dummy_prev_h})
    print('Simplified LSTM Model')
    print('batch_size: '..batch_size..', layers: 2')
    print(y)
    graph.dot(simple_model.fg, 'Simple LSTM')

    alex_model = LSTM.create(1000, 2, 100, 'alex')
    dummy_input  = torch.rand(batch_size, 1000)
    dummy_prev_c = torch.rand(batch_size, 100)
    dummy_prev_h = torch.rand(batch_size, 100)
    y = alex_model:forward({dummy_input, dummy_prev_c, dummy_prev_h, dummy_prev_c, dummy_prev_h})
    print('Alex Grave\'s LSTM Model')
    print('batch_size: '..batch_size..', layers: 2')
    print(y)
    graph.dot(alex_model.fg, 'Alex LSTM')

end

return LSTM




