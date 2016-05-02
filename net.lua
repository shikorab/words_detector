-- requires
require 'nn'

--[[
	Creating saving and loading nets required for detector
--]]
local net = {}
-- Net Utilities
function net:save_model(name, model)
	torch.save("models/" .. name .. ".net" , model:float())
	model:double()
end

function net:load_model(name)
	local model = torch.load("models/" .. name .. ".net")
	return model:double()
end



--Nets
net.net12 = "net12"
function net:create_net12()
	print("Define 12 network")
	---- NETWORK
	
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 48
	local KERNEL_CONV_LAYER1 = {3, 3}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 48
	local FILTERS_CONV_LAYER2 = 2
	local KERNEL_CONV_LAYER2 = {4, 4}

	model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialSoftMax())
	
	--self:save_model(self.net12, model)
	return model:double()
end


net.net12_calib = "net12_calib"
function net:create_net12_calib()
	print("Define 12 calib network")
	---- NETWORK
	
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 128
	local KERNEL_CONV_LAYER1 = {3, 3}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 128
	local FILTERS_CONV_LAYER2 = 28
	local KERNEL_CONV_LAYER2 = {4, 4}

	model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	
	--self:save_model(self.net12, model)
	return model:double()
end


net.net1224 = "net1224"
function net:create_net1224()
	print("Define 1224 network")
	---- NETWORK
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 16
	local KERNEL_CONV_LAYER1 = {5, 3}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 16
	local FILTERS_CONV_LAYER2 = 2
	local KERNEL_CONV_LAYER2 = {9, 4}

	model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	model:add(nn.View(-1))
	model:add(nn.LogSoftMax())
	
	
	--self:save_model(self.net12, model)
	return model:double()
end

net.net24 = "net24"
function net:create_net24()
	---- 24 NETWORK
	print("Define 24 net")
	
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 128
	local KERNEL_CONV_LAYER1 = {5, 5}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 128
	local FILTERS_CONV_LAYER2 = 2
	local KERNEL_CONV_LAYER2 = {9, 9}

	local model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialSoftMax())
	
	--self:save_model(self.net24, model)
	return model
end

net.net48 = "net48"
function net:create_net48()
	---- 48 NETWORK
	print("Define 48 net")
	
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 128
	local KERNEL_CONV_LAYER1 = {5, 5}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 128
	local FILTERS_CONV_LAYER2 = 2
	local KERNEL_CONV_LAYER2 = {17, 17}

	local model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(FILTERS_CONV_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialSoftMax())
	
	--self:save_model(self.net48, model)
	return model
end


net.net48fixed = "net48fixed"
function net:create_net48fixed()
	---- 48 NETWORK
	print("Define 48 net")
	
	local INPUT_CHANNELS_LAYER1 = 1
	local FILTERS_CONV_LAYER1 = 128
	local KERNEL_CONV_LAYER1 = {5, 5}
	local KERNEL_POOL = {3, 3}
	local STRIDE_POOL = {2, 2}
	local INPUT_CHANNELS_LAYER2 = 128
	local FILTERS_CONV_LAYER2 = 2
	local KERNEL_CONV_LAYER2 = {8, 8}

	local model = nn.Sequential()
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(FILTERS_CONV_LAYER1, FILTERS_CONV_LAYER1, KERNEL_CONV_LAYER1[1], KERNEL_CONV_LAYER1[2]))
	model:add(nn.SpatialMaxPooling(KERNEL_POOL[1], KERNEL_POOL[2], STRIDE_POOL[1], STRIDE_POOL[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(INPUT_CHANNELS_LAYER2, FILTERS_CONV_LAYER2, KERNEL_CONV_LAYER2[1], KERNEL_CONV_LAYER2[2]))
	model:add(nn.ReLU())
	model:add(nn.SpatialSoftMax())
	
	--self:save_model(self.net48, model)
	return model
end

return net
