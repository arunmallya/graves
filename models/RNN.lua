--[[ 
This implements two types of RNNs: 
1) 'alex' -  http://arxiv.org/pdf/1308.0850v5.pdf, Page 3
    layer 1 has access to prev_h[1] and x_t
    layer L(>1) has access to prev_h[L], next_h[L-1] and x_t
2) 'simplified' - similar to https://github.com/karpathy/char-rnn/blob/master/model/RNN.lua
    layer 1 has access to prev_h[1] and x_t
    layer L(>1) has access to prev_h[L] and next_h[L-1]
    
batch_size >= 1 is supported by this code
--]]

require 'nngraph'

local RNN = {}

function RNN.create(input_size, num_L, num_h, output_size, rnn_type)

    -- check input sanity
    rnn_type = rnn_type or 'simplified'
    assert(rnn_type == 'simplified' or rnn_type == 'alex', 
        'invalid rnn type: '..rnn_type)
    assert(input_size > 0, 'invalid input_size: should be > 0')
    assert(output_size > 0, 'invalid output_size: should be > 0')
    assert(num_L >= 1, 'invalid num_L: should be >= 1')
    if not torch.isTensor(num_h) then 
        num_h = torch.Tensor(num_L):fill(num_h)
    else
        assert(num_h:size(1) == num_L,
            [[invalid num_h: need one scalar or 
            a tensor of same length as num_L]])
    end

    print('Creating '..rnn_type..' RNN of '..num_L..' layers of size')
    print(num_h)

    -- there are 1+n inputs: x_t, prev_h[1], ..., prev_h[n]
    local inputs = {}
    table.insert(inputs, nn.Identity()():annotate{'x'})
    for L = 1, num_L do
        table.insert(inputs, nn.Identity()():annotate{'prev_h_'..L})
    end

    -- build the computation graph of each layer
    local outputs = {}
    local x = inputs[1]
    local input, next_h, input_size_L, rnn_size
    local num_hidden = 0
    for L = 1, num_L do
        rnn_size = num_h[L]
        num_hidden = num_hidden + rnn_size

        -- inputs for this layer
        if L == 1 then
            -- layer takes x input
            input = x
            input_size_L = input_size
        else
            if rnn_type == 'simplified' then 
                -- layer takes next_h[L-1] as input
                input = outputs[L-1]
                input_size_L = num_h[L-1]
            elseif rnn_type == 'alex' then
                -- layer takes x, next_h[L-1] as input
                input = nn.JoinTable(2)({x, outputs[L-1]}) -- [x, next_h[L-1]]
                input_size_L = num_h[L-1] + input_size
            end
        end
        local prev_h = inputs[1+L]

        -- matrix multiplication
        local i2h = nn.Linear(input_size_L, rnn_size)(input):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h_'..L}

        -- form the output
        next_h = nn.Tanh()(nn.CAddTable()({i2h, h2h})):annotate{name='next_h_'..L}

        table.insert(outputs, next_h)
    end
       
    -- set up the decoder
    if rnn_type == 'simplified' then 
        local top_h = outputs[#outputs]
        local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
        local logsoft = nn.LogSoftMax()(proj)
        table.insert(outputs, logsoft)
    elseif rnn_type == 'alex' then
        local top_h = nn.JoinTable(2)(outputs)
        local proj = nn.Linear(num_hidden, output_size)(top_h):annotate{name='decoder'}
        local logsoft = nn.LogSoftMax()(proj)
        table.insert(outputs, logsoft)
    end

    -- create a module out of the computation graph
    return nn.gModule(inputs, outputs)

end

function RNN.test()
-- create RNN, do a forward pass and plot the forward computation graph

    batch_size = 2

    model = RNN.create(1000, 1, 100, 500)
    dummy_input = torch.rand(batch_size, 1000)
    dummy_prev  = torch.rand(batch_size, 100)
    y = model:forward({dummy_input, dummy_prev})
    print('Single Layer RNN Model')
    print('batch_size: '..batch_size..', layers: 1')
    print(y)
    graph.dot(model.fg, 'Single Layer RNN')

    simple_model = RNN.create(1000, 2, 100, 500)
    dummy_input = torch.rand(batch_size, 1000)
    dummy_prev  = torch.rand(batch_size, 100)
    y = simple_model:forward({dummy_input, dummy_prev, dummy_prev})
    print('Simplified RNN Model')
    print('batch_size: '..batch_size..', layers: 2')
    print(y)
    graph.dot(simple_model.fg, 'Simple RNN')

    alex_model = RNN.create(1000, 2, torch.Tensor{100, 200}, 500, 'alex')
    dummy_input = torch.rand(batch_size, 1000)
    dummy_prev1 = torch.rand(batch_size, 100)
    dummy_prev2 = torch.rand(batch_size, 200)
    y = alex_model:forward({dummy_input, dummy_prev1, dummy_prev2})
    print('Alex Grave\'s RNN Model')
    print('batch_size: '..batch_size..', layers: 2')
    print(y)
    graph.dot(alex_model.fg, 'Alex Grave\'s RNN')

end

return RNN




