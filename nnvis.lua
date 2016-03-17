require "qt"
require "qtwidget"
require "torch"
require "nn"

-- qtwidget.newwindow() sample

function main()--network)
	local inputs = 3
	local outputs = 1
	local HUs = 10
	network = nn.Sequential();
	network:add(nn.Linear(inputs, HUs))
	network:add(nn.Tanh())
	network:add(nn.Linear(HUs, outputs))
	network:add(nn.Tanh())
	network:forward(torch.rand(inputs)) -- Some dummy data to test
	local width = 800
	local height = 600
	local w = qtwidget.newwindow(width,height,"Neural Network Visualization")
	local painter = w.port
	local n_layers = (#network.modules)+1
	local column_half_width = torch.floor(width/(2*n_layers))


	local function reset()
		print("Function reset not yet implemented")
	end

	local function get_node_color(value,range)
		--TODO: other colormaps?
		local color_low = torch.Tensor({0.3,0.3,1})
		local color_high = torch.Tensor({1,0.3,0.3})
		if range[1] == range[2] then --If no distribution
			return unpack(torch.totable(color_low))
		end
		local color = color_low+(color_high-color_low)*(value-range[1])/(range[2]-range[1])
		return unpack(torch.totable(color))
	end

	local function get_link_color(value,range)
		local color_zero = torch.Tensor({1,1,1}) -- links with 0 weight will be white
		local color_low = torch.Tensor({0.3,0.3,1})
		local color_high = torch.Tensor({1,0.3,0.3})
		if range[1] == range[2] then --If no distribution
			return unpack(torch.totable(color_high))
		end
		local abs_max = torch.max(torch.abs(torch.Tensor(range)))
		local color
		if value > 0 then
			color = color_zero+(color_high-color_zero)*(value)/(abs_max)
		else
			color = color_zero+(color_low-color_zero)*(-value)/(abs_max)
		end
		return unpack(torch.totable(color))
	end

	local function draw()
		painter:gbegin()
		painter:showpage()
		painter:gsave()

		local prev_n_nodes
		local n_nodes
		--Draw graph
		local function draw_nodes(layer,layer_type) --layer type can be nil or a value, if it's "nn.Tanh" or "nn.Sigmoid" range values are set manually
			local node_values
			if layer == 0 then
				n_nodes = network.modules[1].weight:size()[2]
				node_values = torch.Tensor(n_nodes):zero() 
			else
				n_nodes = network.modules[layer].output:size()[1]
				node_values = network.modules[layer].output
			end
			local y_spacing = height/(2*n_nodes)
			local y_offset = height/2 - (((n_nodes-1)/2))*y_spacing 
			local node_radius = math.min(column_half_width/2,y_spacing/3)
			local range
			if layer_type == 'nn.Tanh' then
				range = {-1,1}
			elseif layer_type == 'nn.Sigmoid' then
				range = {0,1}
			else
				range = {torch.min(node_values),torch.max(node_values)}
			end
 
			for i = 1,n_nodes do
				-- draw circle
				painter:newpath()
				painter:setcolor("black")
				painter:arc(column_half_width*(2*layer+1),y_offset+(i-1)*y_spacing,node_radius,0,360)
				painter:stroke(false)
				--fill with activation
				painter:setcolor(get_node_color(node_values[i],range))
				painter:fill()
			end	
		end 

		local function draw_node_links(layer)
			assert(layer > 0)
			n_nodes = network.modules[layer].output:size()[1]
			local link_weights
			if network.modules[layer].weight then
				link_weights = network.modules[layer].weight:clone()		
			else
				assert(n_nodes == prev_n_nodes)
				link_weights = torch.eye(n_nodes)
			end
			local y_spacing = height/(2*n_nodes)
			local y_offset = height/2 - (((n_nodes-1)/2))*y_spacing 
			local prev_y_spacing = height/(2*prev_n_nodes)
			local prev_y_offset = height/2 - (((prev_n_nodes-1)/2))*prev_y_spacing 
			local range = {torch.min(link_weights),torch.max(link_weights)}
 
			for i = 1,prev_n_nodes do
				for j = 1,n_nodes do
					if link_weights[j][i] ~= 0 then
						--draw link
						painter:newpath()
						painter:setcolor(get_link_color(link_weights[j][i],range))
						painter:moveto(column_half_width*(2*layer-1),prev_y_offset+(i-1)*prev_y_spacing)
						painter:lineto(column_half_width*(2*layer+1),y_offset+(j-1)*y_spacing,node_radius)
						painter:stroke()
					end
				end
			end	
			prev_n_nodes = n_nodes
		end 
	
		--Initial value
		prev_n_nodes = network.modules[1].weight:size()[2]	
		for layer = 1,(n_layers-1) do
			draw_node_links(layer)
		end
		for layer = 0,(n_layers-1) do
			draw_nodes(layer,tostring(network.modules[layer]))
		end
		painter:grestore()
		painter:gend()
	end

	local function click(x,y)
		print("clicked:",x,y)
	end

	local function key(k,n)
		if k:tostring() == "r" then
			reset()
			draw()
		end
	end

	-- mouse event
	qt.connect(w.listener,
		"sigMousePress(int,int,QByteArray,QByteArray,QByteArray)",
		function(x,y) click(x,y) end)

	-- keyboard event
	qt.connect(w.listener,
		"sigKeyPress(QString,QByteArray,QByteArray)",
		function(k,n) key(k,n) end)

	reset()
	draw()
end

main()
