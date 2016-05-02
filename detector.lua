-- requires
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'
dofile('nms.lua')

local detector = {}


--detector parameters 
detector.net12_threshold = 0.5
detector.net24_threshold = 0.5 -- masked
detector.net48_threshold = 0.5

detector.scale_overlap_threshold = 0.2
detector.total_overlap_threshold = 0.2


detector.hscales = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
detector.wscales = {0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13} 

--defines
detector.net12_size = 12
detector.net24_size = 24
detector.net48_size = 48


-- load the required nets for detector
function detector:load(net)
	self.net12 = net:load_model(net.net12)
	self.net24 = net:load_model(net.net24)
	self.net48 = net:load_model(net.net48)
end

--[[
	Anaylze image
		Return detection database
		image_detection = {
			words = {
			[i] = {[1] = <x1>, [2] = <y1>, [3] = <x2>, [4] = <y2>}
			}
		}
--]]
function detector:forward(img)
	
	image_detection = {}
	
	-- If gray images - duplicate channel
	local size = img:size()
	local channel, width, height = size[1], size[2], size[3]
	-- Run 12net
	for wscale = 1 , table.getn(self.wscales) do
		for hscale = 1 , table.getn(self.hscales) do
			
			local width = img:size(2) * self.wscales[wscale]
			local height = img:size(3) * self.hscales[hscale]
			local scaled_im = image.scale(img,height ,width)
			local actual_scale_width = scaled_im:size(2) / img:size(2)
			local actual_scale_height = scaled_im:size(3) / img:size(3)
			
			--forward
			local res = self.net12:forward(scaled_im)
	
			-- Extract positives and run 24net
			local pos_box = res[1]:gt(self.net12_threshold)
	
			local num_of_boxes = pos_box:sum()
			print(string.format("hscale %1.2f,wscale %1.2f - options = %d",  self.hscales[hscale], self.wscales[wscale], res:size(2) * res:size(3)))
			print(string.format("hscale %1.2f,wscale %1.2f - 12net detections = %d", self.hscales[hscale], self.wscales[wscale], num_of_boxes))
			local boxes = torch.DoubleTensor(num_of_boxes, 5):zero()
			local box_index = 1
			for w = 1, res:size(2) - 1 do
				for h = 1, res:size(3) - 1 do
					if pos_box[w][h] == 1 then
					
						-- positive over 12 net - run over 24 net to filter false postive
						-- crop the relevant 24 net
						local x1 = ((w - 1) * 2 + 1) / actual_scale_width 
						local y1 = ((h - 1) * 2 + 1) / actual_scale_height
						local x2 = ((self.net12_size - 1) / actual_scale_width) + x1
						local y2 = ((self.net12_size - 1) / actual_scale_height) + y1
						
						local window = image.crop(img, y1, x1, y2, x2)
						-- local img24 = image.scale(window, self.net24_size, self.net24_size)
						-- local img24_res = self.net24:forward(img24)
						
						-- if img24_res[1][1][1] > self.net24_threshold then
							
							
						local img48 = image.scale(window, self.net48_size, self.net48_size)
						local img48_res = self.net48:forward(img48)
					
						if img48_res[1][1][1] > self.net48_threshold then
							local width_fix = x2 - x1
							local height_fix = y2 - y1
							x1 = x1 + 0.14 * (width_fix)
							x2 = x2 - 0.14 * (width_fix)
							y1 = y1 + 0.14 * (height_fix)
							y2 = y2 - 0.14 * (height_fix)
				
			
							boxes[box_index][1] = x1
							boxes[box_index][2] = y1
							boxes[box_index][3] = x2
							boxes[box_index][4] = y2
							boxes[box_index][5] = img48_res[1][1][1]
							box_index = box_index + 1
						end
						--end
					end
				end
			end
			
			print(string.format("hscale %1.2f,wscale %1.2f - 48net detections = %d", self.hscales[hscale], self.wscales[wscale], box_index - 1))
			
			--Run NMS over each scale
			if box_index > 1 then
				boxes = boxes[{{1, box_index - 1}, {}}]
				pick_boxes = nms(boxes, self.scale_overlap_threshold)
				
				print(string.format("hscale %1.2f,wscale %1.2f - nms detections = %d", self.hscales[hscale], self.wscales[wscale], pick_boxes:numel()))
				
				for q= 1, pick_boxes:numel() do
					table.insert(image_detection, boxes[pick_boxes[q]])
				end
			end
			print("")
			
		end
	end
	
	
	
	print(string.format("total before nms  = %d",  table.getn(image_detection)))
	
	-- Running NMS over all scales
	boxes = torch.DoubleTensor(table.getn(image_detection), 5):zero()
	for q = 1, table.getn(image_detection) do
		box = image_detection[q]
		boxes[q] = box
	end
	pick_boxes = nms(boxes, self.total_overlap_threshold)
	print(string.format("total after nms  = %d",  pick_boxes:numel()))
	
	local image_detection = {}
	for q= 1, pick_boxes:numel() do
		table.insert(image_detection, boxes[pick_boxes[q]])
	end
	
	return image_detection
end


--Given detection database and tagged words data base
--Save image with the bounding box marked (green - tagged, red - detected)
function detector:display_results(im, dwords, twords, file_name)

	local img = im
	if im:size(1) < 3 then
		img = torch.DoubleTensor(3, im:size(2), im:size(3))
		img[1] = im[1]
		img[2] = im[1]
		img[3] = im[1]
	end
	
	for word_index=1, table.getn(dwords) do
		local word = dwords[word_index]
		if word[1] > 0 and word[3] > 0 then
			
			--Fill rectungle
			j = word[2]
			for i = word[1], word[3] do
				img[1][i][j] = 255
				img[2][i][j] = 0
				img[3][i][j] = 0
			end
			
			j = word[4]
			for i = word[1], word[3] do
				img[1][i][j] = 255
				img[2][i][j] = 0
				img[3][i][j] = 0
			end
			
			i = word[1]
			for j = word[2], word[4] do
				img[1][i][j] = 255
				img[2][i][j] = 0
				img[3][i][j] = 0
			end
			i = word[3]
			for j = word[2], word[4] do
				img[1][i][j] = 255
				img[2][i][j] = 0
				img[3][i][j] = 0
			end
		end
	end
	
	for word_index=1, table.getn(twords) do
		local word = twords[word_index]
		if word.x1 > 0 and word.x2 then
			local width_fix = word.x2 - word.x1
			local height_fix = word.y2 - word.y1
			local x1 = word.x1 + 0.14 * (width_fix)
			local x2 = word.x2 - 0.14 * (width_fix)
			local y1 = word.y1 + 0.14 * (height_fix)
			local y2 = word.y2 - 0.14 * (height_fix)
			--Fill rectungle
			j = y1
			for i = x1, x2 do
				img[1][i][j] = 0
				img[2][i][j] = 255
				img[3][i][j] = 0
			end
			
			j = y2
			for i = x1, x2 do
				img[1][i][j] = 0
				img[2][i][j] = 255
				img[3][i][j] = 0
			end
			
			i = x1
			for j = y1, y2 do
				img[1][i][j] = 0
				img[2][i][j] = 255
				img[3][i][j] = 0
			end
			i = x2
			for j = y1, y2 do
				img[1][i][j] = 0
				img[2][i][j] = 255
				img[3][i][j] = 0
			end
		end
	end
	image.saveJPG(file_name, img)
	--image.display(img)
end

--Itererate over all images and and create neg_words data base with negatives with IOU < <threshold>
function detector:images_iterate(words, threshold, start_index, end_index)
	local neg_words = {}
	neg_words.images = {}
	neg_words.words_count = 0
	for im = start_index, end_index do
		local image_info = words.images[im]
		local img = image.load(image_info.image)
		local res = self:forward(img)
		local detect_neg_info = eval:eval_image(res, image_info, threshold)
		table.insert(neg_words.images, detect_neg_info)
		neg_words.words_count = neg_words.words_count + table.getn(detect_neg_info.words) 
	end
	
	return neg_words
end
return detector