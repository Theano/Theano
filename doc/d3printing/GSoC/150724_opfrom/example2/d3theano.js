function flipAxes(nodes) {
	var size = [0, 0];
	for (var i in nodes) {
		var node = nodes[i];
		size[0] = Math.max(size[0], node.pos[0] + node.width);
		size[1] = Math.max(size[1], node.pos[1] + node.height);
	}	
	for (var i in nodes) {
		var node = nodes[i];
		node.pos[1] = size[1] - (node.pos[1] + node.height);
	}
}


function processDotGraph(dotGraph) {
	dotGraph.rnodes = {};
	for (var nodeId in dotGraph._nodes) {
		var node = dotGraph._nodes[nodeId];
		node.id = nodeId;
		node.isCluster = nodeId.startsWith('cluster');
		if (!node.isCluster) {
			dotGraph.rnodes[nodeId] = node;
		}
	}
	
	var i = 0;
	for (var nodeId in dotGraph.rnodes) {
		var node = dotGraph._nodes[nodeId];
		node.pos = node.pos.split(',').map(function(d) {return parseInt(d);});
		var size = textSize(node.label, {'class': 'nodeText'});
		node.width = size.width + 2 * pad;
		node.height = size.height + 2 * pad;
		node.cx = node.width / 2;
		node.cy = node.height / 2;
		node.hasChilds = exists(node.subg);
		node.showChilds = false;
		node.profile = parseProfile(node.profile);
		if (node.profile.length) {
			isProfiled = true;
		}
		if (exists(node.subg_map_inputs)) {
			node.subg_map_inputs = eval(node.subg_map_inputs)
		}
		if (exists(node.subg_map_outputs)) {
			node.subg_map_outputs = eval(node.subg_map_outputs)
		}
	}
	
	flipAxes(dotGraph.rnodes);
	
	// Offset and scale positions
	var posMin = [Infinity, Infinity];
	for (var i in dotGraph.rnodes) {
		var node = dotGraph._nodes[i];
		posMin[0] = Math.min(posMin[0], node.pos[0]);
		posMin[1] = Math.min(posMin[1], node.pos[1]);
	}
	for (var i in dotGraph.rnodes) {
		var node = dotGraph._nodes[i];
		var pos = node.pos;
		pos[0] -= posMin[0];
		pos[1] -= posMin[1];
		pos[0] = 1.2 * pos[0];
		pos[1] = 1.2 * pos[1];
	}
	
	var edges = dotGraph.edges();
	for (var i in edges) {
		var edge = dotGraph.edge(edges[i]);
		var size = textSize(edge.label, {'class': 'edgeLabelText'});
		edge.width = size.width + 2 * pad;
		edge.height = size.height + 2 * pad;
		if (!exists(edge.color)) {
			edge.color = 'black';
		}
		switch (edge.color) {
			case 'dodgerblue':
				edge.type = 'b';
				break;
			case 'red':
				edge.type = 'r';
				break;
			default:
				edge.type = 'n';
		}
	}
}


function traverseChilds(dotGraph, nodes, groups, parent) {
	var preId = '';
	var ref = undefined;
	var group = {'id': groups.length, 'nodes': [], 'parent': parent};
	if (exists(parent)) {
		ref = parent.value.subg;
		group.parent = parent;
		group.nodes.push(parent);
		parent.group = group;
	}
	groups.push(group);
	var childs = dotGraph.children(ref);
	for (var i in childs) {
		var child = dotGraph.node(childs[i]);
		if (child.isCluster) {
			continue;
		}
		var node = {
			'id': child.id,
			'value': child,
			'index': nodes.length,
			'fixed': true,
			'group': group,
			'isParent': child.showChilds,
			'parent': parent
			};
		nodes.push(node);
		if (child.showChilds) {
			traverseChilds(dotGraph, nodes, groups, node);
		} else {
			group.nodes.push(node);
		}
	}
}

function groupSize(nodes) {
	var minPos = [Infinity, Infinity];
	var maxPos = [-Infinity, -Infinity];
	for (var i in nodes) {
		var node = nodes[i];
		if (node.isParent) {
			continue;
		}
		minPos[0] = Math.min(minPos[0], node.value.pos[0]);
		minPos[1] = Math.min(minPos[1], node.value.pos[1]);
		maxPos[0] = Math.max(maxPos[0], node.value.pos[0] + node.value.width);
		maxPos[1] = Math.max(maxPos[1], node.value.pos[1] + node.value.height);
	}
	return [maxPos[0] - minPos[0], maxPos[1] - minPos[1]];
}

function forceGraph(dotGraph, prevGraph) {
	var graph = {'nodes': [], 'groups': []};
	traverseChilds(dotGraph, graph.nodes, graph.groups);
	
	graph.nodesd = {};
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		graph.nodesd[node.id] = node;
	}
	
	graph.nodesp = graph.nodes.filter(function(d) {return d.isParent;});
	graph.nodesn = graph.nodes.filter(function(d) {return !d.isParent;});
	
	for (i in graph.groups) {
		var group = graph.groups[i];
		group.size = groupSize(group.nodes);
		var parent = group.parent;
		if (exists(parent)) {
			var prevParent = prevGraph.nodesd[group.parent.id];
			if (exists(prevParent)) {
				group.pos = [prevParent.x, prevParent.y];
			} else {
				group.pos = parent.pos.slice(0);
			}
			group.pos[0] += parent.value.cx;
			group.pos[1] += parent.value.cy;
		} else {
			group.pos = [group.size[0] / 2, group.size[1] / 2];
		}
		var min = [Infinity, Infinity];
		for (var j in group.nodes) {
			var node = group.nodes[j];
			if (!node.isParent) {
				min[0] = Math.min(min[0], node.value.pos[0]);
				min[1] = Math.min(min[0], node.value.pos[1]);
			}
		}
		for (var j in group.nodes) {
			var node = group.nodes[j];
			if (node.isParent) {
				node.x = group.pos[0];
				node.y = group.pos[1];
			} else {
				node.x = group.pos[0] - group.size[0] / 2 + node.value.pos[0] - min[0];
				node.y = group.pos[1] - group.size[1] / 2 + node.value.pos[1] - min[1];
			}
		}
	}
	
	graph.size = graph.groups[0].size;
	
	// Reuse previous positions
	if (exists(prevGraph)) {
		for (var i in graph.nodes) {
			var node = graph.nodes[i];
			var prevNode;
			prevNode = prevGraph.nodesd[node.id];
			if (exists(prevNode)) {
				node.x = prevNode.x;
				node.y = prevNode.y;
				node.fixed = prevNode.fixed;
			} else {
				for (var j in prevGraph.groups) {
					var group = prevGraph.groups[j];
					if (exists(group.parent) && group.parent.id == node.id) {
						node.x = group.pos[0] + group.size[0] / 2;
						node.y = group.pos[1] + group.size[1] / 2;
					}
				}
			}
		}
	}
	
	// Edges
	graph.edges = [];
	
	for (var i in graph.nodesn) {
		for (var j in graph.nodesn) {
			var source = graph.nodesn[i];
			var target = graph.nodesn[j];
			
			var dotEdge = dotGraph.edge(source.value.id, target.value.id);
			if (exists(dotEdge)) {
				var edge = {};
				edge.source = parseInt(source.index);
				edge.target = parseInt(target.index);
				edge.value = dotEdge;
				graph.edges.push(edge);
			}
			
			function redirectEdges(map, dotEdge) {
				for (var k in map) {
					var kmap = map[k];
					if (kmap[0] == source.id && kmap[1] == target.id) {
						var edge = {};
						edge.source = parseInt(source.index);
						edge.target = parseInt(target.index);
						edge.value = dotEdge;
						graph.edges.push(edge);	
					}
				}
			}

			var map = undefined;
			if (exists(target.parent)) {
				var parent = target.parent;
				var dotEdge = dotGraph.edge(source.id, parent.id);
				if (exists(dotEdge)) {
					map = parent.value.subg_map_inputs;
					redirectEdges(map, dotEdge);
				}
			}
			
			if (exists(source.parent)) {
				var parent = source.parent;
				var dotEdge = dotGraph.edge(parent.id, target.id);
				if (exists(dotEdge)) {
					map = parent.value.subg_map_outputs;
					redirectEdges(map, dotEdge);
				}
			}
		}
	}

	return graph;
}

function convexHulls(graph, offset) {
	var hulls = [];
	offset = offset || 20;
	for (var i in graph.groups) {
		var group = graph.groups[i];
		if (!exists(group.parent)) {
			continue;
		}
		var points = [];
		for (var j in group.nodes) {
			var node = group.nodes[j];
			if (node.isParent) {
				points.push([node.x, node.y]);
			} else {
				points.push([node.x - node.value.cx - offset, node.y - node.value.cy - offset]);
				points.push([node.x - node.value.cx - offset, node.y + node.value.cy + offset]);
				points.push([node.x + node.value.cx + offset, node.y - node.value.cy - offset]);
				points.push([node.x + node.value.cx + offset, node.y + node.value.cy + offset]);	
			}
		}
		hulls.push({group: i, path: d3.geom.hull(points)});
	}
	return hulls;
}

function drawCluster(d) {
  return curve(d.path); // 0.8
}

function setupGraph() {
	if (isProfiled) {
		d3.select('body').select('#menu').append('input')
			.attr('name', 'tColors')
			.attr('type', 'button')
			.attr('value', 'Toggle profile colors')
			.attr('onclick', "toggleColors()");
			
		maxProfilePer = 0;
		for (i in graph.nodes) {
			var p = graph.nodes[i].value.profile;
			if (p.length) {
				maxProfilePer = Math.max(maxProfilePer, p[0] / p[1]);
			}
		}
	}
	
	var isEdgeOver = false;
	var isEdgeLabelOver = false;
	
	graph.hulls = convexHulls(graph);
	hulls = pane.selectAll('#hulls').remove();
	hulls = pane.append('g').attr('id', 'hulls')
		.selectAll('path')
		.data(graph.hulls).enter()
		.append('path')
		.attr('class', 'hull')
		.attr('d', drawCluster);
		
		
	hulls.on('dblclick', function(d) {
		var parent = graph.groups[d.group].parent;
		parent.value.showChilds = !parent.value.showChilds;
		graph = forceGraph(dotGraph, graph);
		setupGraph();
	});
	
	// Add edges
	edges = pane.selectAll('#edges').remove();
	edges = pane.append('g').attr('id', 'edges')
		.selectAll('path').data(graph.edges).enter().append('path')
		.attr('class', 'edge')
		.attr('stroke', function(d) {return d.value.color;})
		.attr('marker-mid', function(d) { return 'url(#edgeArrow_' + d.value.type + ')';});
		
	edges.on('mouseover', function(d) {
			var edge = d3.select(this);
			edge.transition()
				.duration(200)
				.style('opacity', 1.0);
		    edgeDiv.transition()        
		        .duration(200)      
		        .style('opacity', .9);
		    edgeDiv
		    	.html(d.value.label)  
		        .style('left', (d3.event.pageX) + 'px')     
		        .style('top', (d3.event.pageY - 28) + 'px');    
		});
		
	edges.on('mouseout', function(d) {
			var edge = d3.select(this);
			edge.transition()
				.duration(200)
				.style('opacity', 0.4);
			edgeDiv.transition()
				.duration(200)
				.style('opacity', 0);
				
			});
			
	// Add nodes
	pane.selectAll('#nodes').remove();
	nodes = pane.append('g').attr('id', 'nodes')
		.selectAll('g').data(graph.nodesn).enter().append('g');
	
	updateNodes();
	updateGraph();
	
	nodes.on('dblclick', function(d) {
		if (d.value.hasChilds) {
			d.value.showChilds = !d.value.showChilds;
			graph = forceGraph(dotGraph, graph);
			setupGraph();
		}
	});
		
	nodes.on('mouseover', function(node) {
		// Highlight incoming edges
		edges.each(function (d, i) {
			var edge = d3.select(this);
			if (d.source == node || d.target == node) {
				edge.transition()
					.duration(200)
					.style('opacity', 1.0);
			}
		});
		
		// Show node details if node is not edited as has profiling information
		if (!isEditNode && node.value.profile.length) {
		   	nodeDiv.transition()        
		        .duration(200)      
		        .style('opacity', .9);
		    nodeDiv
		    	.html(nodeDetails(node))  
		        .style('left', (d3.event.pageX) + 30 + 'px')     
		        .style('top', (d3.event.pageY - 28) + 'px');
		}
	});
		
	nodes.on('mouseout', function(node) {
		edges.each(function (d, i) {
			var edge = d3.select(this);
			if (d.source.index == node.index || d.target.index == node.index) {
				edge.transition()
					.duration(200)
					.style('opacity', 0.4);
			}
		});
	   	hideNodeDiv();
	});
	
	nodes.on('contextmenu', d3.contextMenu(menuItems));
	
	// Force layout
	layout = d3.layout.force()
		.nodes(graph.nodes)
		.links(graph.edges)
		.size(graph.size)
		.charge(-300)
		.linkDistance(400)
		.linkStrength(0.4)
		.gravity(0)
		.on('tick', updateGraph);
		
	// Drag behavour
	var drag = layout.drag()
		.on('dragstart', function(d) {
			d3.event.sourceEvent.stopPropagation();
			d3.event.sourceEvent.preventDefault();
			d.fixed = true;
		});
	nodes.call(drag);
		
	// Start force layout
	layout.start();
}

function length(x1, y1, x2, y2) {
	return Math.sqrt(Math.pow(x1-x2, 2) + Math.pow(y1-y2, 2));
}

function pathPos(x1, y1, x2, y2, c) {
	x = (1 - c) * x1 + c * x2;
	y = (1 - c) * y1 + c * y2;
	p = x + ',' + y;
	return p;
}

function updateGraph() {
	graph.hulls = convexHulls(graph);
	hulls.data(graph.hulls)
		.attr('d', drawCluster);
	
	// Update nodes
	nodes.attr('transform', function(d) { return 'translate(' + (d.x - d.value.cx) + ' ' + (d.y - d.value.cy) + ')'; });
	// Update edges
	edges.attr('d', function(d) {
		var dist = 100;
		var l = length(d.source.x, d.source.y, d.target.x, d.target.y);
		var n = Math.max(2, Math.floor(l / dist));
		var marker = [];
		for (var i = 1; i < n; ++i) {
			marker.push(i / n);
		}
		var markerPos = marker.map(function(c) {return pathPos(d.source.x, d.source.y, d.target.x, d.target.y, c);});
		var markerPos = ' L' + markerPos.join(' L');
		return 'M' + d.source.x + ',' + d.source.y + markerPos + ' L' + d.target.x + ',' + d.target.y;
	});
}
		
function toggleColors() {
		colorProfile = !colorProfile;
		updateNodes();
		updateGraph();
	}
		
function textSize(text, attr) {
		var t = svg.append('text').text(text);
	if (typeof(attr) != 'undefined') {
		for (a in attr) {
			t.attr(a, attr[a]);
		}
	}
	var bbox = t.node().getBBox();
	t.remove();
	return bbox;
}

function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}

function exists(x) {
	return typeof(x) != 'undefined';
}

function replaceAll(str, find, replace) {
	return str.replace(new RegExp(find, 'g'), replace);
}

function parseProfile(profile) {
	if (typeof(profile) == 'undefined') {
		return [];
	}
	profile = profile.replace('[', '');
	profile = profile.replace(']', '');
	profile = replaceAll(profile, ' ', '');
	profile = profile.split(',');
	if (profile.length < 2) {
		return [];
	}
	profile = profile.map(function(x) { return parseFloat(x); });
	return profile;
}

function linspace(start, end, len) {
	var d = (end - start) / (len - 1);
	var rv = [start];
	for (i = 1; i < len; ++i) {
		rv.push(rv[i - 1] + d);
	}
	return rv;
}

function profileColor(per) {
	var s = d3.scale.linear()
		.domain(linspace(0, maxProfilePer, profileColors.length))
		.range(profileColors)
		.interpolate(d3.interpolateRgb);
	return s(per);
}


function fillColor(d) {
	if (colorProfile && d.value.profile.length) {
		if (d.value.shape == 'ellipse') {
			return profileColor(d.value.profile[0] / d.value.profile[1]);
		} else {
			return 'white';
		}
	} else {
		return typeof(d.value.fillcolor) == 'undefined' ? 'white' : d.value.fillcolor;
	}
}

function formatTime(sec) {
	var s;
	if (sec < 0.1) {
		s = (sec * 1000).toFixed(1) + 'ms';
	} else {
		s = sec.toFixed(1) + 's';
	}
	return s;
}

function nodeDetails(node) {
	var s = '<b>' + node.value.label + '</b>';
	var p = node.value.profile;
	if (p.length) {
		s += '<br>Time: ' + formatTime(p[0]);
		s += '<br>Time: ' + (p[0] / p[1] * 100).toFixed(1) + '%';
	}
	return s;	
}

function updateNode(d, node) {
	var shape;
	if (d.value.shape == 'ellipse') {
		node.selectAll('ellipse').remove();
		shape = node.append('ellipse')
			.attr('class', 'nodeEllipse')
			.attr('cx', d.value.cx)
			.attr('cy', d.value.cy)
			.attr('rx', d.value.width * 0.6)
			.attr('ry', d.value.height * 0.6);

	} else {
		node.selectAll('rect').remove();
		shape = node.append('rect')
			.attr('class', 'nodeRect')
			.attr('width', d.value.width)
			.attr('height', d.value.height);
	}
	shape.attr('fill', fillColor(d));
	
	node.selectAll('text').remove();
	var text = node.append('text')
		.attr('class', 'nodeText')
		.attr('x', pad)
		.attr('dy', function(d) {return d.value.height - pad - 5;})
		.text(function(d) {return d.value.label;});
		
	if (d.value.hasChilds) {
		node.style('cursor', 'pointer');
	}
}

function updateNodes() {
	nodes.each(function(d) {
		var node = d3.select(this);
		updateNode(d, node);
	});	
}

function hideNodeDiv() {
	nodeDiv.transition()        
        .duration(200)      
        .style('opacity', 0);
}

function setNodeSize(node) {
	var size = textSize(node.value.label, {'class': 'nodeText'});
		node.value.width = size.width + 2 * pad;
		node.value.height = size.height + 2 * pad;
		node.value.cx = node.value.width / 2;
		node.value.cy = node.value.height / 2;
	}
	
function editNode(elm, d) {
		var node = d3.select(elm);
		var pos = elm.getBBox();
		if (d3.event.defaultPrevented) return;
		
		isEditNode = true;
		hideNodeDiv();
		
		var form = node.append('foreignObject')
		.attr('x', pos.x)
		.attr('y', pos.y)
		.attr('width', d.value.width)
		.attr('height', 25);
	var input = form.append('xhtml:form').append('input')
		.attr('style', 'width: ' + d.value.width + 'px')
		.attr('value', function() {
				this.focus();
				return d.value.label;
		})
		.on('blur', function() {
			d.value.label = input.node().value;
			setNodeSize(d);
			updateNode(d, node);
			form.remove(); // TODO: check this
			isEditNode = false;
		})
		.on('keypress', function() {
			if (!d3.event) {
				d3.event = window.event;
			}
			var event = d3.event;
			if (event.keyCode == 13) {
				if (typeof(event.cancelBubble)) {
					event.cancelBubble = true;
				}
				if (event.stopPropagation) {
					event.stopPropagation();
				}
				event.preventDefault();
				d.value.label = input.node().value;
				setNodeSize(d);
				updateNode(d, node);
				form.remove(); // TODO: check this
				isEditNode = false;
			}
		});
}

function releaseNode(d) {
	d.fixed = false;
	layout.start();
}

function releaseNodes() {
	graph['nodes'].forEach (function (d) {
		d.fixed = false;
	});
	layout.start();
}

function resetNodes() {
	layout.stop();
	var nodes = graph['nodes'];
	nodes.forEach(function (node, i){
		nodes[i].x = scaleDotX(node.value.pos[0]);
		nodes[i].y = scaleDotY(dotGraph.values.height - (node.value.pos[1] + node.value.height));
		nodes[i].px = nodes[i].x;
		nodes[i].py = nodes[i].y;
		nodes[i].fixed = true;
	});
	updateGraph();
	layout.start();
}
