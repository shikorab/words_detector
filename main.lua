--[[
	Training 12-net
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")

model = net:load_model(net.net12)
--model = net:create_net12()
--model:reset()
-- model:remove(6)
-- model:add(nn.View(-1))
-- model:add(nn.LogSoftMax())

--words = data:read_train_data()
-- ata:save_data(data.train_data_name, words)
words = data:load_data(data.train_data_name)
tensor_data = data:make_data_tensor(words, 1, 12 ,12)

--test_words = data:read_test_data()
-- data:save_data(data.test_data_name, test_words)
test_words = data:load_data(data.test_data_name)
test_tensor_data = data:make_data_tensor(test_words, 1, 12 ,12)

labels = torch.DoubleTensor(128, 2, 1, 1):zero()
labels:indexFill(2,torch.LongTensor({1}),1)
for i=0, (128/4- 1) do
	labels[i * 4 + 4][1][1][1] = 0
	labels[i * 4 + 4][2][1][1] = 1
end

-- labels_class = torch.DoubleTensor(4):zero()
-- labels_class[1] = 2
-- labels_class[2] = 2
-- labels_class[3] = 2
-- labels_class[4] = 1

training.OptimState = {
	nesterov = true ,
	learningRate = 0.01,
	learningRateDecay = 0,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0.9 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 100
training:train(net.net12, model, tensor_data, test_tensor_data, labels)
-------------------------------------------------------------------------------



--[[
	Training 12-net calib
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")

model = net:create_net12_calib()
words = data:load_data(data.train_data_name)
tensor_data = data:make_calib_data_tensor(words, 1, 12 ,12)

--test_words = data:read_test_data()
-- data:save_data(data.test_data_name, test_words)
test_words = data:load_data(data.test_data_name)
test_tensor_data = data:make_calib_data_tensor(test_words, 1, 12 ,12)

labels = torch.DoubleTensor(140, 28, 1, 1):zero()
for i=1, 140 do
	labels[i][(i - 1) % 28 + 1][1][1] = 1
end

training.OptimState = {
	nesterov = true ,
	learningRate = 0.01,
	learningRateDecay = 0,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0.9 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 100
training:train(net.net12_calib, model, tensor_data, test_tensor_data, labels)
----------------------------------------------------------------------------


--[[
	Training 24-net
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")

model = net:load_model(net.net24)
--model = net:create_net24()
--model:reset()

--words = data:read_train_data()
--data:save_data(data.train_data_name, words)
words = data:load_data(data.train_data_name)
tensor_data = data:make_random_data_tensor(words, 1, 24 ,24)

--test_words = data:read_test_data()
--data:save_data(data.test_data_name, test_words)
test_words = data:load_data(data.test_data_name)
test_tensor_data = data:make_random_data_tensor(test_words, 1, 24 ,24)

labels = torch.DoubleTensor(128, 2, 1, 1):zero()
labels:indexFill(2,torch.LongTensor({1}),1)
for i=0, (128/4- 1) do
	labels[i * 4 + 4][1][1][1] = 0
	labels[i * 4 + 4][2][1][1] = 1
end
-- labels_class = torch.DoubleTensor(4):zero()
-- labels_class[1] = 2
-- labels_class[2] = 2
-- labels_class[3] = 2
-- labels_class[4] = 1

training.OptimState = {
	nesterov = true ,
	learningRate = 0.01,
	learningRateDecay = 0,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0.3 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 1000
model = training:train(net.net24, model, tensor_data, test_tensor_data, labels)



--[[
	Training 48-net
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")

model = net:load_model(net.net48)
--model = net:create_net48()
--model:reset()

words = data:load_data(data.train_data_name)
tensor_data = data:make_random_data_tensor(words, 1, 48 ,48, 25600, 1)

-- test_words = data:read_test_data()
-- data:save_data(data.test_data_name, test_words)
test_words = data:load_data(data.test_data_name)
test_tensor_data = data:make_random_data_tensor(test_words, 1, 48 ,48, 1280, 1)

labels = torch.DoubleTensor(64, 1, 2):zero()
labels:indexFill(3,torch.LongTensor({1}),1)
for i=0, (64/4- 1) do
	labels[i * 4 + 2][1][1] = 0
	labels[i * 4 + 2][1][2] = 1
	labels[i * 4 + 3][1][1] = 0
	labels[i * 4 + 3][1][2] = 1
	labels[i * 4 + 4][1][1] = 0
	labels[i * 4 + 4][1][2] = 1
end


training.OptimState = {
	nesterov = true ,
	learningRate = 0.01,
	learningRateDecay = 0,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0.3 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 100
training:train(net.net48, model, tensor_data, test_tensor_data, labels)
------------------------------------------------------------------------------

--[[
	Analyze image
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")
detector = dofile("detector.lua")
eval = dofile("eval.lua")

detector:load(net)
words = data:load_data(data.test_data_name)
--words = data:read_train_data()
im = image.load(words.images[5].image)
image_detection = detector:forward(im)
image_detection.image = words.images[5].image
detector:display_results(im, image_detection, words.images[5].words)
neg_words = eval:eval_image(image_detection, words.images[5], 0.6)


--[[
	Analyze test benchmark
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")
detector = dofile("detector.lua")
eval = dofile("eval.lua")

detector:load(net)
words = data:load_data(data.test_data_name)
--words = data:read_train_data()
count = 0
relevant = 0
selected = 0
for imi = 1, table.getn(words.images) do
	im = image.load(words.images[imi].image)
	image_detection = detector:forward(im)
	image_detection.image = words.images[imi].image
	detector:display_results(im, image_detection, words.images[imi].words, "img"..imi..".jpg")
	info = eval:eval_image(image_detection, words.images[imi], 0.6)
	count = count + info.count
	relevant = relevant + info.relevant
	selected = selected + info.selected
	print("image %d", imi)
	print(string.format("sum true positive %d", count))
	print(string.format("sum false positive %d", selected - count))
	print(string.format("sum false negative %d", relevant - count))
	print(string.format("sum recall %f", count/relevant))
	print(string.format("sum precision %f", count/selected))
end

--[[
	Run lior's benchmark
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")
detector = dofile("detector.lua")
eval = dofile("eval.lua")

detector:load(net)
im = image.load("lior/1.jpg")
img = torch.DoubleTensor(1, im:size(2), im:size(3)):zero()
img[1] = torch.add(torch.add(im[1], im[2]), im[3])
img = torch.div(img, 3)
img = img:gt(0.5)
img = img:type("torch.DoubleTensor")
image_detection = detector:forward(img)
image_detection.image = " "
detector:display_results(im, image_detection, {})
--neg_words = eval:eval_image(image_detection, words.images[5], 0.6)

--[[
	Iterative Training for 48-net
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("net.lua")
detector = dofile("detector.lua")
eval = dofile("eval.lua")

detector:load(net)
words = data:load_data(data.train_data_name)
-- pos_tensor = data:make_pos_data_tensor(words, 1, 48, 48)
-- neg_words = detector:images_iterate(words, 0.6, 1, 40)
-- neg_tensor = data:make_neg_data_tensor(neg_words, 1, 48, 48, words)
-- data_tensor = data:combine_data_tensor(pos_tensor, neg_tensor)
-- torch.save("data_tensor", data_tensor)
data_tensor =  torch.load("data_tensor")
test_tensor = data_tensor[{{1, 32}, {}}]

labels = torch.DoubleTensor(32, 1, 2):zero()
labels:indexFill(3,torch.LongTensor({1}),1)
for i=1, (32/2) do
	labels[i * 2][1][1] = 0
	labels[i * 2][1][2] = 1
end

training.OptimState = {
	nesterov = true ,
	learningRate = 0.0001,
	learningRateDecay = 0,
	learningRateMul = 1,
	learningRateCount = 1,
	momentum = 0.3 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 100
model = net:load_model(net.net48.."new")
--model = net:create_net48fixed()
training:train(net.net48.."new", model, data_tensor, test_tensor, labels)




--[[
	Training 1224-net
--]]
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'

training = dofile("training.lua")
data = dofile("data.lua")
net = dofile("models/net.lua")

model = net:load_model(net.net1224)
--model = net:create_net1224()
--model:reset()

-- words = data:read_train_data()
-- ata:save_data(data.train_data_name, words)
words = data:load_data(data.train_data_name)
tensor_data = data:make_data_tensor(words, 1, 12 ,24)

-- test_words = data:read_test_data()
-- data:save_data(data.test_data_name, test_words)
test_words = data:load_data(data.test_data_name)
test_tensor_data = data:make_data_tensor(test_words, 1, 12 ,24)

-- labels = torch.DoubleTensor(2):zero()
-- labels[1] = 2
-- labels[2] = 1


labels = torch.DoubleTensor(64, 1, 2):zero()
labels:indexFill(3,torch.LongTensor({1}),1)
for i=0, (64/2- 1) do
	labels[i * 2 + 1][1][1] = 0
	labels[i * 2 + 1][1][2] = 1
end


training.OptimState = {
	nesterov = true ,
	learningRate = 0.0001,
	learningRateDecay = 0,
	learningRateCount = 1,
	learningRateMul = 1,
	momentum = 0.3 ,
	dampening = 0 ,
	-- weightDecay = 0.05
}
training.epochs = 3
model = training:train(net.net1224.."test", model, tensor_data, test_tensor_data, labels)
