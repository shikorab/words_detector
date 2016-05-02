--[[
  Convert Data to required format
  1. Reads the benchmark and create bounding box
  2. Adding xframe and yframe word length to the bounding box.
  3. Support random manipulations for both positive data and negative data
  4. Addtional data manipulations utilities
--]]

require 'image'

local data = {}

-- image per word
-- Holds the last read created words database
data.train_words = {}
data.test_words = {}

--Path to benchmark
data.train_images_path = "data/images_train"
data.train_tagged_images_path = "data/gt_words_train"
data.test_images_path = "data/images_test"
data.test_tagged_images_path = "data/gt_words_test"

--environment to include in word bounding box 
data.xframe = 0.25
data.yframe = 0.25

--create words bounding box database for train data
function data:read_train_data()
	words = self:read_data(self.train_images_path, self.train_tagged_images_path, self.train_words)
	collectgarbage("count")
	return words
end

--create words bounding box database for test data
function data:read_test_data()
	words = self:read_data(self.test_images_path, self.test_tagged_images_path, self.test_words)
	collectgarbage("count")
	return words
end

--[[
	internal function to actully create the bounding box database
		images_path - path to original images location (expecting ".jpg" format)
		tagged_path - path to original tagged images (expeting ICDAR 2013 format)
		words -	(output) words data base
				Format:
				words = {
					images = {
						[i] = {
							image = <image full path>
							words = {x1, y1, x2, y2}
						}
					}
					words_count = <total number of words in database>
				}
--]]
function data:read_data(images_path, tagged_path, words)

	words.images = {}	
	words.words_count = 0
			
	for file in paths.files(images_path) do
		-- We only load files that match the extension
		local image_path
		local im
		local file_im
		local image_tagged_path
		local tagged_im
		local tagged_tensor
		local image_info
		if file:find('.jpg') then
			-- read image and tagged image
			image_path = paths.concat(images_path, file)
			im = image.loadJPG(image_path)
			print(file)
			
			
			file_im = string.gsub(file, "jpg", "tif.dat")
			image_tagged_path = paths.concat(tagged_path, file_im)

			
			tagged_im = torch.IntStorage(image_tagged_path)
			tagged_tensor = torch.IntTensor(tagged_im)
			tagged_tensor = tagged_tensor:reshape(im:size())
			
			
			image_info = {}
			image_info.image = image_path
			image_info.words = {}
			self:read_tagged_words(im, tagged_tensor, image_info.words)
			collectgarbage("count")
			table.insert(words.images, image_info)
			words.words_count = words.words_count + table.getn(image_info.words)
			print(words.words_count)
			
			

		end
	end
	
	return words
	
end
--reading image tagged data and adding the words ro database (see read_data) 
function data:read_tagged_words(im, tagged_tensor, words)
	local nof_words = tagged_tensor:max()
	local mini = torch.IntTensor(nof_words):fill(tagged_tensor:size(2))
	local minj = torch.IntTensor(nof_words):fill(tagged_tensor:size(3))
	local maxi = torch.IntTensor(nof_words):zero()
	local maxj = torch.IntTensor(nof_words):zero()
	local t_tagged_tensor = torch.data(tagged_tensor)
	local width = tagged_tensor:size(3)
	for i = 0, tagged_tensor:size(2) - 1 do
		for j = 0, tagged_tensor:size(3) - 1 do
			local elem = t_tagged_tensor[i*width + j]
			if elem > 0 then				
				if mini[elem] > i  then
					mini[elem] = i
				end
				if minj[elem] > j then
					minj[elem] = j
				end
				if maxi[elem] < i then
					maxi[elem] = i
				end
				if maxj[elem] < j then
					maxj[elem] = j
				end
			end
		end
	end
	
	
	for word = 1, nof_words do
		if maxi[word] > 0  then
			local word_info = {}
			local xframe = (maxi[word] - mini[word]) * self.xframe
			local orig_xframe = xframe 
			xframe = math.min(xframe, mini[word])
			xframe = math.min(xframe, tagged_tensor:size(2) - maxi[word] - 1)
			local yframe = (maxj[word] - minj[word]) * self.yframe
			local orig_yframe = yframe
			yframe = math.min(yframe, minj[word])
			yframe = math.min(yframe, tagged_tensor:size(3) - maxj[word] - 1)
			
			if (yframe == orig_yframe) and (xframe == orig_xframe) then
				word_info.x1 = mini[word] + 1 - xframe
				word_info.y1 = minj[word] + 1 - yframe
				word_info.x2 = maxi[word] + 1 + xframe
				word_info.y2 = maxj[word] + 1 + yframe
				-- local window = image.crop(im, minj[word] + 1, mini[word] + 1, maxj[word] + 1, maxi[word] + 1)
				table.insert(words, word_info)
			end
		end
	end
	
end

--save and load words database
data.train_data_name = "train_data"
data.test_data_name = "test_data"
function data:save_data(name, db)
	torch.save("data/" .. name .. ".dat" , db)
end

function data:load_data(name)
	local db = torch.load("data/" .. name .. ".dat")
	return db
end


--[[
	Getting words data base (with format created in function "read_data") 
	and create data tensor with both positive data and negative data.
	words - words data base
	cahnnels - checked just with 1
	width, height - require size (i.e. <12, 12>, <48, 48> .. )
	max_nof_words - stop after this number of words
	more_negs - add this parameter to switch from 1 negative for 3 positives to 3 negatives for 1 positive
--]]
function data:make_data_tensor(words, channels, width, height, max_nof_words, more_negs)
	
	local nof_words
	if more_negs~=nil then
		nof_words = math.floor(words.words_count / 3) * 4 -- one negative for 3 positives
	else 
		nof_words = words.words_count* 4 -- 3 negative for 1 positive
	end
	if max_nof_words ~= nil then
		nof_words = math.min(nof_words, max_nof_words)
	end
	local data_tensor = torch.DoubleTensor(nof_words, channels, width, height):zero()
	
	local index = 1
	local x1 = 0
	local y1 = 0
	local x2 = 0 
	local y2 = 0

	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			if index > nof_words then
				break
			end
			local word_info = image_info.words[word_index]
			
			-- add one positive and 1 negative (out of 7)
			-- add the word
			local window = image.crop(img, word_info.y1, word_info.x1, word_info.y2, word_info.x2)
			data_tensor[index][1]:copy(image.scale(window, width, height))
			if (channels == 3) then
				data_tensor[index][3] = data_tensor[index][1]
				data_tensor[index][2] = data_tensor[index][1]
			end
			index = index + 1
			
			local add
			if more_negs == nil then
				add = ((index - 1) % 24) == 3
			else
				add = (word_index % 2) == 0
			end
			
			if add then 
				-- add shift left word
				local window = image.crop(img, word_info.y1, (word_info.x1 + word_info.x2) / 2 , word_info.y2, word_info.x2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			if more_negs == nil then
				add = ((index - 1) % 24) == 7
			else
				add = (word_index % 2) == 0
			end
			if add then
				-- add shift right word
				local window = image.crop(img, word_info.y1, word_info.x1, word_info.y2, (word_info.x1 + word_info.x2) / 2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			if more_negs == nil then
				add = ((index - 1) % 24) == 11
			else
				add = (word_index % 2) == 0
			end
			if add then
				-- add shift up word
				local window = image.crop(img, (word_info.y1 + word_info.y2) / 2, word_info.x1,  word_info.y2, word_info.x2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			if more_negs == nil then
				add = ((index - 1) % 24) == 15
			else
				add = (word_index % 2) == 1
			end
			if add then
				-- add shift down word
				local window = image.crop(img, word_info.y1, word_info.x1, (word_info.y1 + word_info.y2) / 2, word_info.x2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			if more_negs == nil then
				add = ((index - 1) % 24) == 19
			else
				add = (word_index % 2) == 1
			end
			if add then
				-- add bigger window
				x1 = math.max(word_info.x1 - (word_info.x2 - word_info.x1) / 2, 1)
				y1 = math.max(word_info.y1 - (word_info.y2 - word_info.y1) / 2, 1)
				x2 = math.min(word_info.x2 + (word_info.x2 - word_info.x1) / 2, img:size(2))
				y2 = math.min(word_info.y2 + (word_info.y2 - word_info.y1) / 2, img:size(3))

				local window = image.crop(img, y1, x1, y2, x2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			if more_negs == nil then
				add = ((index - 1) % 24) == 23
			else
				add = (word_index % 2) == 1
			end
			if add then
				-- add smaller window
				x1 = word_info.x1 + (word_info.x2 - word_info.x1) / 4
				y1 = word_info.y1 + (word_info.y2 - word_info.y1) / 4
				x2 = word_info.x2 - (word_info.x2 - word_info.x1) / 4
				y2 = word_info.y2 - (word_info.y2 - word_info.y1) / 4
				
				local window = image.crop(img, y1, x1, y2, x2)
				data_tensor[index][1]:copy(image.scale(window, width, height))
				if (channels == 3) then
					data_tensor[index][3] = data_tensor[index][1]
					data_tensor[index][2] = data_tensor[index][1]
				end
				index = index + 1
			end
			
			-- if (index % 7) == 6 then
				-- -- add random window
				-- data_tensor[index] = torch.rand(channels, height, width)
				-- if (channels == 3) then
					-- data_tensor[index][3] = data_tensor[index][1]
					-- data_tensor[index][2] = data_tensor[index][1]
				-- end
				-- index = index + 1
			-- end
			
		end
		if index > nof_words then
			break
		end
	end
	
	return data_tensor
end

-- See "make_data_tensor" for more details
-- (additnal random manipulations added to both negative and positive data)
function data:make_random_data_tensor(words, channels, width, height, max_nof_words, more_negs)
	
	local nof_words
	if more_negs~=nil then
		nof_words = math.floor(words.words_count / 3) * 4 -- one negative for 3 positives
	else 
		nof_words = words.words_count* 4 -- 3 negative for 1 positive
	end
	if max_nof_words ~= nil then
		nof_words = math.min(nof_words, max_nof_words)
	end
	local data_tensor = torch.DoubleTensor(nof_words, channels, width, height):zero()
	
	local index = 1
	local x1 = 0
	local y1 = 0
	local x2 = 0 
	local y2 = 0

	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			if index > nof_words then
				break
			end
			local word_info = image_info.words[word_index]
			
			-- add one positive and 1 negative (out of 7)
			-- add the word
			local window = image.crop(img, word_info.y1, word_info.x1, word_info.y2, word_info.x2)
			data_tensor[index][1]:copy(image.scale(window, width, height))
			index = index + 1
			
			local add
			local count
			if more_negs == nil then
				add = ((index - 1) % 4) == 3
				count = 1
			else
				add = 1
				count = 3
			end
			
			if add then
				for add_index = 1, count do
					while true do
						local x1  = math.random(img:size(2) - 1)
						local x2  = x1 + math.random(math.min(img:size(2) - x1, 2 * (word_info.x2 - word_info.x1)))
						local y1  = math.random(img:size(3) - 1)
						local y2  = y1 + math.random(math.min(img:size(3) - y1, 2 * (word_info.y2 - word_info.y1)))
						
						local iou_max = 0
						for tagged_word_index = 1, table.getn(image_info.words) do
							local tword = image_info.words[tagged_word_index]
							
							-- Calc IOU
							local max_x1 = math.max(x1, tword.x1)
							local min_x2 = math.min(x2, tword.x2)
							if max_x1 < min_x2 then
								local deltax = min_x2 - max_x1
								
								local max_y1 = math.max(y1, tword.y1)
								local min_y2 = math.min(y2, tword.y2)
							
								if max_y1 < min_y2 then
									local deltay = min_y2 - max_y1
									
									local intersection = deltax * deltay
									local union = (x2 - x1) * (y2 - y1) + (tword.x2 - tword.x1) * (tword.y2 - tword.y1) - intersection 
									
									local iou = intersection / union
									if iou > iou_max then
										iou_max = iou
									end
								end
							end	
						end
			
		
						if iou_max < 0.6 then
							local window = image.crop(img, y1, x1, y2, x2)
							data_tensor[index][1]:copy(image.scale(window, width, height))
							index = index + 1
							break
						end
					
					end
				end
			end
		end	
			
		if index > nof_words then
			break
		end
	end
	
	return data_tensor
end

--Create data for calib net
function data:make_calib_data_tensor(words, channels, width, height)
	
	nof_words = words.words_count * 28
	local data_tensor = torch.DoubleTensor(nof_words, channels, width, height):zero()
	
	local index = 1
	
	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			local word_info = image_info.words[word_index]
			
			
			local word_width = word_info.x2 - word_info.x1
			local word_height = word_info.y2 - word_info.y1
			
			local skip = 0
			
			if word_info.x1 - 0.15 * word_width < 1 then
				skip = 1
			end
			if word_info.y1 - 0.15 * word_height < 1 then
				skip = 1
			end
			if word_info.x2 + 0.15 * word_width > img:size(2) then
				skip = 1
			end
			if word_info.y2 + 0.15 * word_height > img:size(3) then
				skip = 1
			end
			
			if skip == 0 then
				for class_shift=1, 28 do
					local x1 = word_info.x1
					local y1 = word_info.y1
					local x2 = word_info.x2
					local y2 = word_info.y2
					
					local shift_index = ((class_shift - 1) % 7) - 3
					if class_shift <= 7 then
						--shift x1
						x1 = x1 + shift_index * 0.05 * word_width
					elseif class_shift <= 14 then
						x2 = x2 + shift_index * 0.05 * word_width
					elseif class_shift <= 21 then
						y1 = y1 + shift_index * 0.05 * word_height
					else
						y2 = y2 + shift_index * 0.05 * word_height
					end
					-- print(x1)
					-- print(x2)
					-- print(y1)
					-- print(y2)
					-- print(img:size())
					local window = image.crop(img, y1, x1, y2, x2)
					data_tensor[index][1]:copy(image.scale(window, width, height))
					index = index + 1
				end
				-- break
			end
		end
		-- break
	end
	
	data_tensor = data_tensor[{{1, index}, {}}]
	return data_tensor
end

--Create postive data only
-- Add max 0.05 word length error
function data:make_pos_data_tensor(words, channels, width, height, max_nof_words)
	
	local nof_words 
	nof_words = words.words_count
	if max_nof_words ~= nil then
		nof_words = math.min(nof_words, max_nof_words)
	end
	local data_tensor = torch.DoubleTensor(nof_words, channels, width, height):zero()
	
	local index = 1

	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			if index > nof_words then
				break
			end
			local word_info = image_info.words[word_index]
			local word_width = word_info.x2 - word_info.x1
			local word_height = word_info.y2 - word_info.y1
			
			local x1 = word_info.x1 + math.random(0.1 * word_width) - 0.05 * word_width
			local x2 = word_info.x2 + math.random(0.1 * word_width) - 0.05 * word_width
			local y1 = word_info.y1 + math.random(0.1 * word_height) - 0.05 * word_height
			local y2 = word_info.y2 + math.random(0.1 * word_height) - 0.05 * word_height
			
			x1 = math.max(x1, 1)
			y1 = math.max(y1, 1)
			x2 = math.min(x2, img:size(2))
			y2 = math.min(y2, img:size(3))
				
				
			local window = image.crop(img, y1, x1, y2, x2)
			data_tensor[index][1]:copy(image.scale(window, width, height))
			index = index + 1

		end
		if index > nof_words then
			break
		end
	end
	
	return data_tensor
end

--Create random negative data
--words - negative words database
--twords - tagged words database (to add additional random negative data with IOU < 0.6)  
function data:make_neg_data_tensor(words, channels, width, height, twords)
	
	local nof_words = 2 * words.words_count
	local data_tensor = torch.DoubleTensor(nof_words, channels, width, height):zero()
	
	local index = 1

	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			if index > nof_words then
				break
			end
			local word_info = image_info.words[word_index]
			local word_width = word_info.x2 - word_info.x1
			local word_height = word_info.y2 - word_info.y1
			
			local x1 = word_info.x1
			local x2 = word_info.x2
			local y1 = word_info.y1
			local y2 = word_info.y2
			
				
			local window = image.crop(img, y1, x1, y2, x2)
			data_tensor[index][1]:copy(image.scale(window, width, height))
			index = index + 1
			
		end
		if index > nof_words then
			break
		end
	end
	
	
	print("start random negs")
	print(index)
	for im = 1, table.getn(words.images) do
		local image_info = words.images[im]
		local img = image.loadJPG(image_info.image)
		for word_index = 1, table.getn(image_info.words) do
			if index > nof_words then
				break
			end
			local word_info = image_info.words[word_index]
			
			while true do
				local x1  = math.random(img:size(2) - 1)
				local x2  = x1 + math.random(math.min(img:size(2) - x1, 2 * (word_info.x2 - word_info.x1)))
				local y1  = math.random(img:size(3) - 1)
				local y2  = y1 + math.random(math.min(img:size(3) - y1, 2 * (word_info.y2 - word_info.y1)))
				
				local iou_max = 0
				for tagged_word_index = 1, table.getn(image_info.words) do
					local tword = image_info.words[tagged_word_index]
					
					-- Calc IOU
					local max_x1 = math.max(x1, tword.x1)
					local min_x2 = math.min(x2, tword.x2)
					if max_x1 < min_x2 then
						local deltax = min_x2 - max_x1
						
						local max_y1 = math.max(y1, tword.y1)
						local min_y2 = math.min(y2, tword.y2)
					
						if max_y1 < min_y2 then
							local deltay = min_y2 - max_y1
							
							local intersection = deltax * deltay
							local union = (x2 - x1) * (y2 - y1) + (tword.x2 - tword.x1) * (tword.y2 - tword.y1) - intersection 
							
							local iou = intersection / union
							if iou > iou_max then
								iou_max = iou
							end
						end
					end	
				end


				if iou_max < 0.6 then
					local window = image.crop(img, y1, x1, y2, x2)
					data_tensor[index][1]:copy(image.scale(window, width, height))
					index = index + 1
					break
				end
			
			end
		end
	end
	return data_tensor
end

--Create 1 to 1 mixed data tensor
function data:combine_data_tensor(pos, neg)
	local length = math.min(neg:size(1), pos:size(1))
	local data_tensor = torch.DoubleTensor(length * 2, pos:size(2), pos:size(3), pos:size(4)):zero()
	for i=1, length do
		data_tensor[(i - 1) * 2 + 1] = pos[i]
		data_tensor[i * 2] = neg[i]
	end
	return data_tensor
end

return data
