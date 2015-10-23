--[[ 
this implements two types of RNNs: 
1) 'alex' -  http://arxiv.org/pdf/1308.0850v5.pdf, Page 3
	layer 1 has access to prev_h[1] and x_t
	layer L(>1) has access to prev_h[L], prev_h[L-1] and x_t
2) 'simplified' - similar to https://github.com/karpathy/char-rnn/blob/master/model/RNN.lua
	layer 1 has access to prev_h[1] and x_t
	layer L(>1) has access to prev_h[L], prev_h[L-1]
]]--

require 'nngraph'

local RNN = {}

function RNN.create(input_size, num_L, num_h, rnn_type)

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

	-- there are 1+n inputs: x_t, prev_h[1], ..., prev_h[n]
	local inputs = {}
	table.insert(inputs, nn.Identity()():annotate{'x'}) -- x_t
	for L = 1, num_L do
		table.insert(inputs, nn.Identity()():annotate{'prev_h_'..L}) -- prev_h[L]
	end

	-- build the computation graph of each layer
	local outputs = {}
	local x = inputs[1]
	local input, next_h, input_size_L
	for L = 1, num_L do

		-- inputs for this layer
		if L == 1 then
			input = x
			input_size_L = input_size
		else
			input = outputs[L-1]
			input_size_L = num_h[L-1]
		end
		local prev_h = inputs[1+L]

		-- matrix multiplication
		local i2h = nn.Linear(input_size_L, num_h[L])(input):annotate{name='i2h_'..L}
		local h2h = nn.Linear(num_h[L], num_h[L])(prev_h):annotate{name='h2h_'..L}

		if L == 1 or rnn_type == 'simplified' then 
		-- layer takes input, prev_h[L] as input
			next_h = nn.Tanh()(nn.CAddTable()({i2h, h2h})):annotate{name='next_h_'..L}
		else
		-- layer takes input, prev_h[L], x_t as input
			local x2h = nn.Linear(input_size, num_h[L])(x):annotate{name='x2h_'..L}
			next_h = nn.Tanh()(nn.CAddTable()({i2h, h2h, x2h})):annotate{name='next_h_'..L}
		end

		table.insert(outputs, next_h)
	end
		
	-- create a module out of the computation graph
	return nn.gModule(inputs, outputs)

end

model = RNN.create(1000, 2, torch.Tensor{100, 200}, 'alex')

dummy_input = torch.rand(1000)
dummy_prev1 = torch.rand(100)
dummy_prev2 = torch.rand(200)

y = model:forward({dummy_input, dummy_prev1, dummy_prev2})
print(y)




