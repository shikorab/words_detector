-- requires
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

--[[
	Utilities for net training
--]]


training = {}

training.epochs = 400


-- Default hyper parameters
training.OptimState = {
	nesterov = false ,
	learningRate = 0.0001,
	learningRateDecay = 0.00003,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}

--criterion for "train function"
training.criterion = nn.MSECriterion()

--Train <model>
--<model> will be saved during the trainig in file "models/<model_name>.net"
--traindata - tenosr with the data
--labels -- tenosr for mini_batch labels (assuming the labes equals per each mini_batch)
--test_data -- for tracking the loss per epoch
function training:train(model_name, model, traindata, testdata, labels)
	-- Set Cost function
	-- local criterion = nn.ClassNLLCriterion()
	logger = optim.Logger("logger"..model_name..".log")
	local mini_batch_size = labels:size(1)
	local batch_size = traindata:size(1)
	local iterations = batch_size / mini_batch_size
	local parameters,gradParameters = model:getParameters()
	
	trainlosses = {}
	testlosses = {}
	
	train_loss, train_err = self:evaluate(model, traindata, labels)
	print(string.format("epoch processed (train): %6s, loss = %6.6f, err = %6.6f", 0, train_loss, train_err))
	trainlosses[#trainlosses + 1] = train_loss -- append the new loss
	best = train_loss
	prev_train_loss = train_loss
		
	test_loss, test_err = self:evaluate(model, testdata, labels)
	print(string.format("epoch processed (test): %6s, loss = %6.6f, err = %6.6f", 0, test_loss, test_err))
	testlosses[#testlosses + 1] = test_loss -- append the new loss
	
	--Logger plot
	logger:add{['% train loss']    = train_loss, ['% test loss']    = test_loss}
	logger:style{['% train loss']    = '-', ['% test loss']    = '-'}
	logger:plot()
	
	
	for epoch=1,self.epochs do
		print(self.OptimState.learningRate)
		for iter=0, iterations - 1 do
			
			local func = function(x)
				local startindex = iter * mini_batch_size + 1
				local endindex = (iter + 1) * mini_batch_size 
				
				local input = traindata[{{startindex, endindex}, {}}]
				local output = model:forward(input)

				-- forward your network and criterion
				-- f should hold the criterion value
				local f = self.criterion:forward(output, labels)

				-- reset gradients
				parameters,gradParameters = model:getParameters()
				gradParameters:zero()

				-- add code to back propagate your network
				model:backward(input, self.criterion:backward(model.output, labels))
				model:updateParameters(self.OptimState.learningRate)
				
				
				return f,gradParameters
			end
			local _, loss = optim.sgd(func, parameters, self.OptimState)
		end
		
		train_loss, train_err = self:evaluate(model, traindata, labels)
		print(string.format("epoch processed (train): %6s, loss = %6.6f, err = %6.6f", epoch, train_loss, train_err))
		trainlosses[#trainlosses + 1] = train_loss -- append the new loss

		
		test_loss, test_err = self:evaluate(model, testdata, labels)
		print(string.format("epoch processed (test): %6s, loss = %6.6f, err = %6.6f", epoch, test_loss, test_err))
		testlosses[#testlosses + 1] = test_loss -- append the new loss
		
		-- Logger plot
		logger:add{['% train loss']    = train_loss, ['% test loss']    = test_loss}
		logger:style{['% train loss']    = '-', ['% test loss']    = '-'}
		logger:plot()
		if train_loss < best then
			print("model saved")
			net:save_model(model_name, model)
			best = train_loss
		end
		if train_loss > prev_train_loss then
			print("model reloaded")
			model = net:load_model(model_name)
			self.OptimState.learningRate = self.OptimState.learningRate / 2
			train_loss = best
		end
		prev_train_loss = train_loss
		if epoch % self.OptimState.learningRateCount == 0 then 
			self.OptimState.learningRate = self.OptimState.learningRate * self.OptimState.learningRateMul
		end
	end
end

--Tracking the loss and calssification error
function training:evaluate(model, eval_data, labels)
	local batch_size = eval_data:size(1)
	local mini_batch_size = labels:size(1)
	local iterations = batch_size / mini_batch_size
	local total_loss = 0
	local match_count = 0
	for iter=0, iterations - 1 do
		local startindex = iter * mini_batch_size + 1
		local endindex = (iter + 1) * mini_batch_size 
				
		local input = eval_data[{{startindex, endindex}, {}}]
		
		local outputs = model:forward(input)
		
		-- compute the loss of these outputs, measured against the true labels in batch_target
		local loss = self.criterion:forward(outputs, labels)
		total_loss = total_loss + loss
		
		-- Calc classification error
		outputs = outputs:gt(0.5):type('torch.DoubleTensor')
		match_count = match_count + outputs:eq(labels):sum()
	end
	
	local err = 1 - match_count/(2 * iterations * mini_batch_size)
	local loss = total_loss / iterations
	
	return loss, err
end


--See "train" function for additional details
--use NLL criterion
function training:train_class(model_name, model, traindata, testdata, labels)
	-- Set Cost function
	local criterion = nn.ClassNLLCriterion()
	logger = optim.Logger("logger_class"..model_name..".log")
	local parameters,gradParameters = model:getParameters()
	local mini_batch_size = labels:size(1)
	local batch_size = traindata:size(1)
	local iterations = batch_size / mini_batch_size
	
	trainlosses = {}
	testlosses = {}
	
	train_loss, train_err = self:evaluate_class(model, criterion, traindata, labels)
	print(string.format("epoch processed (train): %6s, loss = %6.6f, err = %6.6f", 0, train_loss, train_err))
	trainlosses[#trainlosses + 1] = train_loss -- append the new loss
	best = train_loss
	prev_train_loss = train_loss
		
	test_loss, test_err = self:evaluate_class(model, criterion, testdata, labels)
	print(string.format("epoch processed (test): %6s, loss = %6.6f, err = %6.6f", 0, test_loss, test_err))
	testlosses[#testlosses + 1] = test_loss -- append the new loss
	
	-- Logger plot
	logger:add{['% train loss']    = train_loss, ['% test loss']    = test_loss}
	logger:style{['% train loss']    = '-', ['% test loss']    = '-'}
	logger:plot()
	
	
	for epoch=1,self.epochs do
		print(self.OptimState.learningRate)
		for iter=0, iterations - 1 do
			for mini_batch_index = 1, mini_batch_size do
				local func = function(x)
					local index = iter * mini_batch_size + mini_batch_index
					
					local input = traindata[index]
					local output = model:forward(input)

					-- forward your network and criterion
					-- f should hold the criterion value
					local f = criterion:forward(output, labels[mini_batch_index])

					-- reset gradients
					local parameters,gradParameters = model:getParameters()
					gradParameters:zero()

					-- add code to back propagate your network
					model:backward(input, criterion:backward(model.output, labels[mini_batch_index]	))
					model:updateParameters(self.OptimState.learningRate)
					
					
					return f,gradParameters
				end
				local _, loss = optim.sgd(func, parameters, self.OptimState)
			end
		end
		
		train_loss, train_err = self:evaluate_class(model, criterion, traindata, labels)
		print(string.format("epoch processed (train): %6s, loss = %6.6f, err = %6.6f", epoch, train_loss, train_err))
		trainlosses[#trainlosses + 1] = train_loss -- append the new loss

		
		test_loss, test_err = self:evaluate_class(model, criterion, testdata, labels)
		print(string.format("epoch processed (test): %6s, loss = %6.6f, err = %6.6f", epoch, test_loss, test_err))
		testlosses[#testlosses + 1] = test_loss -- append the new loss
		
		-- Logger plot
		logger:add{['% train loss']    = train_loss, ['% test loss']    = test_loss}
		logger:style{['% train loss']    = '-', ['% test loss']    = '-'}
		logger:plot()
		if train_loss < best then
			print("model saved")
			--net:save_model(model_name, model)
			best = train_loss
		end
		if train_loss > prev_train_loss then
			print("model reloaded")
			--model = net:load_model(model_name)
			self.OptimState.learningRate = self.OptimState.learningRate / 2
			train_loss = best
		end
		prev_train_loss = train_loss
		--self.OptimState.learningRate = self.OptimState.learningRate * self.OptimState.learningRateMul
	end
	
	return model
end

--Tracking loss and calssification error (NLL criterion)
function training:evaluate_class(model, criterion, eval_data, labels)
	local batch_size = eval_data:size(1)
	local mini_batch_size = labels:size(1)
	local iterations = batch_size / mini_batch_size
	local total_loss = 0
	local match_count = 0
	for iter=0, iterations - 1 do
		for mini_batch_index = 1, mini_batch_size do
				
			local index = iter * mini_batch_size + mini_batch_index 
			local input = eval_data[index]
			
			local outputs = model:forward(input)
			
			-- compute the loss of these outputs, measured against the true labels in batch_target
			local loss = criterion:forward(outputs, labels[mini_batch_index])
			total_loss = total_loss + loss
			
			-- Calc classification error
			local target = labels[mini_batch_index]
			local no_target = (target % 2) + 1
			if outputs[target] > outputs[no_target] then 
				match_count = match_count + 1
			end
		end
	end
	
	local err = 1 - match_count/(iterations * mini_batch_size)

	
	return total_loss, err
end


return training
