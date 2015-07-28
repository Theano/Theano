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
	// Merge and remove subgraph nodes
	for (var nodeId in dotGraph._nodes) {
		if (!exists(dotGraph._nodes[nodeId])) {
			continue;
		}
		if (nodeId.startsWith('cluster_')) {
			var id = nodeId.replace('cluster_', '');
			assert(exists(dotGraph._nodes[id]));
			var parent = dotGraph.node(id);
			var childIds = dotGraph.children(nodeId);
			for (var i in childIds) {
				var childId = childIds[i];
				dotGraph.setParent(childId, id);
				dotGraph.setEdge(childId, id, {'label': 'opfrom'});
				var child = dotGraph.node(childId);
			}
			dotGraph.removeNode(nodeId);
		}
	}
	
	var i = 0;
	for (var nodeId in dotGraph._nodes) {
		var node = dotGraph._nodes[nodeId];
		node.id = nodeId;
		node.pos = node.pos.split(',').map(function(d) {return parseInt(d);});
		var size = textSize(node.label, {'class': 'nodeText'});
		node.width = size.width + 2 * pad;
		node.height = size.height + 2 * pad;
		node.cx = node.width / 2;
		node.cy = node.height / 2;
		node.hasChilds = dotGraph.children(nodeId).length > 0;
		node.showChilds = false;
		node.profile = parseProfile(node.profile);
		if (node.profile.length) {
			isProfiled = true;
		}
	}
	
	flipAxes(dotGraph._nodes);
	
	// Offset and scale positions
	var posMin = [Infinity, Infinity];
	for (var i in dotGraph._nodes) {
		var node = dotGraph._nodes[i];
		posMin[0] = Math.min(posMin[0], node.pos[0]);
		posMin[1] = Math.min(posMin[1], node.pos[1]);
	}
	for (var i in dotGraph._nodes) {
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

function makeNode(dotGraph, dotNode) {
	var node = {};
	node.value = dotNode;
	node.fixed = true;
	return node;
}

function traverseChilds(dotGraph, parent) {
	var childs = dotGraph.children(parent);
	var nodes = [];
	for (var i in childs) {
		var child = dotGraph.node(childs[i]);
		nodes.push(makeNode(dotGraph, child));
		if (child.showChilds) {
			nodes = nodes.concat(traverseChilds(dotGraph, child.id));
		}
	}
	return nodes;
}

function forceGraph(dotGraph, prevGraph) {
	// Parse nodes
	var graph = {};
	graph.nodes = traverseChilds(dotGraph);
	
	graph.nodesd = {};
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		node.index = i;
		graph.nodesd[node.value.id] = node;
	}
	
	var groups = {};
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		var parentId = dotGraph.parent(node.value.id);
		if (exists(parentId)) {
			if (!(parentId in groups)) {
				groups[parentId] = [];
			}
			groups[parentId].push(node.value.id);
		}
	}
	
	// Compute group centroids
	var groupsMeta = {};
	for (var i in groups) {
		var group = groups[i];
		var cx = 0, cy = 0;
		for (var j in group) {
			var node = graph.nodesd[group[j]];
			cx += node.value.pos[0];
			cy += node.value.pos[1];
		}
		var n = groups[i].length;
		cx /= n;
		cy /= n;
		groupsMeta[i] = {'cx': cx, 'cy': cy, 'n': n};
	}
	
	// Reuse previous positions
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		var prevNode;
		if (exists(prevGraph)) {
			prevNode = prevGraph.nodesd[node.value.id];
		}
		if (exists(prevNode)) {
			node.x = prevNode.x;
			node.y = prevNode.y;
			node.fixed = prevNode.fixed;
		} else {
			var parentId = dotGraph.parent(node.value.id);
			if (exists(parentId)) {
				var parentPos;
				var parent = prevGraph.nodesd[parentId];
				if (exists(parent)) {
					parentPos = [parent.x, parent.y];
				} else {
					parent = graph.nodesd[parentId];
					parentPos = parent.value.pos;
				}
				var g = groupsMeta[parentId];
				node.x = parentPos[0] + node.value.pos[0] - g.cx;
				node.y = parentPos[1] - 100 + node.value.pos[1] - g.cy;
				node.fixed = true;
			} else {
				node.x = node.value.pos[0];
				node.y = node.value.pos[1];
			}
		}
	}
	
	// Offset graph on initialization
	if (!exists(prevGraph)) {
		var posMin = [Infinity, Infinity];
		for (var i in graph.nodes) {
			var node = graph.nodes[i];
			posMin[0] = Math.min(posMin[0], node.x);
			posMin[1] = Math.min(posMin[1], node.y);
		}
		for (var i in graph.nodes) {
			var node = graph.nodes[i];
			node.x -= posMin[0];
			node.y -= posMin[1];
		}
	}
	
	// Compute dimension of graph
	var minPos = [Infinity, Infinity];
	var maxPos = [-Infinity, -Infinity];
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		minPos[0] = Math.min(minPos[0], node.x);
		minPos[1] = Math.min(minPos[1], node.y);
		maxPos[0] = Math.max(maxPos[0], node.x + node.value.width);
		maxPos[1] = Math.max(maxPos[1], node.y + node.value.height);
	}
	graph.dim = {'minPos': minPos, 'maxPos': maxPos,
	'size': [maxPos[0] - minPos[0], maxPos[1] - minPos[0]]};
	
	// Edges
	graph.edges = [];
	for (var i in graph.nodes) {
		for (var j in graph.nodes) {
			var sourceId = graph.nodes[i].value.id;
			var targetId = graph.nodes[j].value.id;
			var dotEdge = dotGraph.edge(sourceId, targetId);
			if (exists(dotEdge)) {
				var edge = {};
				edge.source = parseInt(graph.nodes[i].index);
				edge.target = parseInt(graph.nodes[j].index);
				edge.value = dotEdge;
				graph.edges.push(edge);
			}
		}
	}
	return graph;
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
		.selectAll('g').data(graph.nodes).enter().append('g');
	
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
		if (!isProfiled || isEditNode || typeof(node.value.profile) == 'undefined') {
			return;
		}
		
		edges.each(function (d, i) {
			var edge = d3.select(this);
			if (d.source == node || d.target == node) {
				edge.transition()
					.duration(200)
					.style('opacity', 1.0);
			}
		});
	   	nodeDiv.transition()        
	        .duration(200)      
	        .style('opacity', .9);
	    nodeDiv
	    	.html(nodeDetails(node))  
	        .style('left', (d3.event.pageX) + 30 + 'px')     
	        .style('top', (d3.event.pageY - 28) + 'px');    
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
		.size(graph.dim.size)
		.charge(-300)
		.linkDistance(300)
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
