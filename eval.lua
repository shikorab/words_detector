-- requires
require 'image'
require 'nn'
require 'optim'
require 'gnuplot'



eval = {}

-- display statstics per classification threhold
function eval:explore(model, data, labels)
	recalls = {}
	
	for threshold=0, 100 do
		true_positives, relevant_elements, selected_elements = self:evaluate(model, data, labels, threshold / 100)
		recall = true_positives/relevant_elements
		print(string.format("threshold %6s, trp = %6.6f, rel = %6.6f, sel=%6.6f, recall=%6.6f ", threshold / 100, true_positives, relevant_elements, selected_elements, recall))
		recalls[#recalls + 1] = reacall	
	end
	
end

-- return statstics for specific classification threhold
function eval:evaluate(model, eval_data, labels, threshold)
	local batch_size = eval_data:size(1)
	local mini_batch_size = labels:size(1)
	local iterations = batch_size / mini_batch_size
	local total_loss = 0
	local match_count = 0
	local relevant_elements = 0
	local selected_elements = 0
	local true_positives = 0
	for iter=0, iterations - 1 do
		local startindex = iter * mini_batch_size + 1
		local endindex = (iter + 1) * mini_batch_size 
				
		local input = eval_data[{{startindex, endindex}, {}}]
		
		outputs = model:forward(input)
		
		-- Calc classification error
		outputs = outputs:gt(threshold):type('torch.DoubleTensor')
		relevant_elements = relevant_elements + labels[{{}, {1, 1}}]:sum()
		selected_elements = selected_elements + outputs[{{}, {1, 1}}]:sum()
		true_positives = true_positives + outputs[{{}, {1, 1}}]:cmul(labels[{{}, {1, 1}}]):sum()
	end
	
	return true_positives, relevant_elements, selected_elements
end

--Create words negative data base for specific image and threshold (given detection database)
function eval:eval_image(detection, image_info, threshold)
	
	local count = 0
	local detect_neg_info = {}
	detect_neg_info.words = {}
	detect_neg_info.image = image_info.image
	
	for detect_word_index = 1, table.getn(detection) do
		local dword = detection[detect_word_index]
		local x1 = dword[1]
		local x2 = dword[3]
		local y1 = dword[2]
		local y2 = dword[4]
						
		local iou_max = 0
	
		for tagged_word_index = 1, table.getn(image_info.words) do
			local tword = image_info.words[tagged_word_index]
			
			local width_fix = tword.x2 - tword.x1
			local height_fix = tword.y2 - tword.y1
			local tx1 = tword.x1 + 0.14 * (width_fix)
			local tx2 = tword.x2 - 0.14 * (width_fix)
			local ty1 = tword.y1 + 0.14 * (height_fix)
			local ty2 = tword.y2 - 0.14 * (height_fix)
		
		
			-- Calc IOU
			local max_x1 = math.max(x1, tx1)
			local min_x2 = math.min(x2, tx2)
			if max_x1 < min_x2 then
				local deltax = min_x2 - max_x1
				
				local max_y1 = math.max(y1, ty1)
				local min_y2 = math.min(y2, ty2)
			
				if max_y1 < min_y2 then
					local deltay = min_y2 - max_y1
					
					local intersection = deltax * deltay
					local union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - intersection 
					
					local iou = intersection / union
					if iou > iou_max then
						iou_max = iou
					end
				end
			end	
		end
		
		
		if iou_max < threshold then
			local word_info = {}
			word_info.x1 = x1
			word_info.y1 = y1
			word_info.x2 = x2
			word_info.y2 = y2
			table.insert(detect_neg_info.words, word_info)
		else	
			count = count + 1
		end
		
	end
	
	print(string.format("true positive %d", count))
	print(string.format("false positive %d", table.getn(detection) - count))
	print(string.format("false negative %d", table.getn(image_info.words) - count))
	print(string.format("recall %f", count/(table.getn(image_info.words))))
	print(string.format("precision %f", count/(table.getn(detection))))
	detect_neg_info.count = count
	detect_neg_info.selected = table.getn(detection)
	detect_neg_info.relevant = table.getn(image_info.words)
	
	return detect_neg_info
end

return eval
