
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function init_cuda(opt)
	-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
	if opt.gpuid >= 0 and opt.opencl == 0 then
	    local ok, cunn = pcall(require, 'cunn')
	    local ok2, cutorch = pcall(require, 'cutorch')
	    if not ok then print('package cunn not found!') end
	    if not ok2 then print('package cutorch not found!') end
	    if ok and ok2 then
	        print('using CUDA on GPU ' .. opt.gpuid .. '...')
	        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
	        cutorch.manualSeed(opt.seed)
	    else
	        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
	        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
	        print('Falling back on CPU mode')
	        opt.gpuid = -1 -- overwrite user setting
	    end
	end

	-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
	if opt.gpuid >= 0 and opt.opencl == 1 then
	    local ok, cunn = pcall(require, 'clnn')
	    local ok2, cutorch = pcall(require, 'cltorch')
	    if not ok then print('package clnn not found!') end
	    if not ok2 then print('package cltorch not found!') end
	    if ok and ok2 then
	        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
	        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
	        torch.manualSeed(opt.seed)
	    else
	        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
	        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
	        print('Falling back on CPU mode')
	        opt.gpuid = -1 -- overwrite user setting
	    end
	end

	return opt
end

-- preprocessing helper function
function prepro(opts, ...)
    local args = {...}
    for i,var in ipairs(args) do
    	if var:dim() == 2 then
        	var = var:transpose(1, 2):contiguous() -- swap the axes for faster indexing
        end
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            var = var:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            var = var:cl()
        end
        args[i] = var
    end
    return unpack(args)
end

