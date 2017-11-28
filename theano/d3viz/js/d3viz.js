/*
 * Theano javascript library for interactive visualization.
 *
 * Author: Christof Angermueller <cangermueller@gmail.com>
*/


/*
 * Checks if variable is defined.
 */
function exists(x) {
	return typeof(x) != 'undefined';
}

/*
 * Replace all patterns in string.
 */
function replaceAll(str, find, replace) {
	return str.replace(new RegExp(find, 'g'), replace);
}


/*
 * Computes len equally spaces points between start and end.
 */
function linspace(start, end, len) {
	var d = (end - start) / (len - 1);
	var rv = [start];
	for (i = 1; i < len; ++i) {
		rv.push(rv[i - 1] + d);
	}
	return rv;
}


/*
 * Converts string to list
 */
function str2List(s) {
	s = s.split('\t');
	return s;
}


/*
 * Flips y-scale such that (0, 0) points to top-left corner.
 */
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


/*
 * Preprocesses raw dotGraph
 */
function processDotGraph(dotGraph) {
	// Ignore cluster nodes
	dotGraph.rnodes = {};
	for (var nodeId in dotGraph._nodes) {
		var node = dotGraph._nodes[nodeId];
		node.id = nodeId;
		node.isCluster = nodeId.substr(0, 7) == 'cluster';
		if (!node.isCluster) {
			dotGraph.rnodes[nodeId] = node;
		}
	}
	
	// Precompute attributes
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
		if (exists(node.profile)) {
			node.profile = parseProfile(node.profile);
			isProfiled = true;
		}
		if (exists(node.tag)) {
			node.tag = str2List(node.tag);
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
	
	// Preprocess edges
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


/*
 * Extracts profiling information from string.
 */
function parseProfile(s) {
	var p = str2List(s);
	p = p.map(function(x) { return parseFloat(x); });
	return p;
}


/*
 * Preprocesses DOT nodes for front-end visualization.
 * Assigns all children of parent (root of graph if not specified)
 * to the same group and calls function recursively on children.
 * 
 */
function traverseChilds(dotGraph, nodes, groups, parent) {
	var preId = '';
	var ref = undefined;
	
	// Create new group with parent as parent
	var group = {'id': groups.length, 'nodes': [], 'parent': parent};
	if (exists(parent)) {
		ref = parent.value.subg;
		group.parent = parent;
		group.nodes.push(parent);
		parent.group = group;
	}
	groups.push(group);
	
	// Loop over all children
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
			'fixed': fixOnInit,
			'group': group,
			'isParent': child.showChilds,
			'parent': parent
			};
		nodes.push(node);
		if (child.showChilds) {
			// Recurse if child is root of subcluster that should be expandend
			traverseChilds(dotGraph, nodes, groups, node);
		} else {
			group.nodes.push(node);
		}
	}
	
	// Groups appended to groups after group are group children.
	group.childs = [];
	for (var i = group.id + 1; i < groups.length; ++i) {
		group.childs.push(groups[i].id);
	}
}


/*
 * Computes width and height of group of nodes.
 */
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


/*
 * Creates front-end graph for visualizing from DOT graph.
 */
function frontEndGraph(dotGraph, prevGraph) {
	var graph = {'nodes': [], 'groups': []};
	traverseChilds(dotGraph, graph.nodes, graph.groups);
	
	// Dictionary to access nodes by id
	graph.nodesd = {};
	for (var i in graph.nodes) {
		var node = graph.nodes[i];
		graph.nodesd[node.id] = node;
	}
	
	// Dictionary to access groups by id
	graph.groupsd = {};
	for (var i in graph.groups) {
		var group = graph.groups[i];
		graph.groupsd[group.id] = group;
	}
	
	// Parent nodes
	graph.nodesp = graph.nodes.filter(function(d) {return d.isParent;});
	// Non-parent nodes
	graph.nodesn = graph.nodes.filter(function(d) {return !d.isParent;});
	
	// Compute size of groups
	for (i in graph.groups) {
		var group = graph.groups[i];
		group.size = groupSize(group.nodes);
		var parent = group.parent;
		if (exists(parent)) {
			var prevParent = prevGraph.nodesd[group.parent.id];
			if (exists(prevParent)) {
				// Restore previous group position if given
				group.pos = [prevParent.x, prevParent.y];
			} else {
				// Use position of parent otherwise
				group.pos = parent.value.pos.slice(0);
			}
			group.pos[0] += parent.value.cx;
			group.pos[1] += parent.value.cy;
		} else {
			group.pos = [group.size[0] / 2, group.size[1] / 2];
		}
		// Offset nodes on group center
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
			if (!node.isParent) {
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
			
			// Redirect edges to subgraph
			
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

/*
 * Computes d3.js convex hull surrounding nodes that
 * belong to the same group.
 */
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
			if (!node.isParent) {
				points.push([node.x - node.value.cx - offset, node.y - node.value.cy - offset]);
				points.push([node.x - node.value.cx - offset, node.y + node.value.cy + offset]);
				points.push([node.x + node.value.cx + offset, node.y - node.value.cy - offset]);
				points.push([node.x + node.value.cx + offset, node.y + node.value.cy + offset]);	
			}
		}
		for (var k in group.childs) {
			var nodes = graph.groupsd[group.childs[k]].nodes;
			for (var j in nodes) {
				var node = nodes[j];
				if (!node.isParent) {
					points.push([node.x - node.value.cx - offset, node.y - node.value.cy - offset]);
					points.push([node.x - node.value.cx - offset, node.y + node.value.cy + offset]);
					points.push([node.x + node.value.cx + offset, node.y - node.value.cy - offset]);
					points.push([node.x + node.value.cx + offset, node.y + node.value.cy + offset]);	
				}
			}
		}
		hulls.push({group: i, path: d3.geom.hull(points)});
	}
	return hulls;
}


/*
 * Draws convex hull.
 */
function drawConvexHull(d) {
	var curve = d3.svg.line()
	    .interpolate("cardinal-closed")
	    .tension(.85);
	return curve(d.path);
}

/*
 * Creates skeleton for graph visualization.
 * Positions will be updated by updateGraph().
 */
function drawGraph() {
	d3.select('body').select('#menu').select('#toggleColors').remove();
	if (isProfiled) {
		d3.select('body').select('#menu').append('input')
			.attr('id', 'toggleColors')
			.attr('type', 'button')
			.attr('value', 'Toggle profile colors')
			.attr('onclick', "toggleNodeColors()");
		maxProfilePer = 0;
		for (i in graph.nodes) {
			var p = graph.nodes[i].value.profile;
			if (exists(p)) {
				maxProfilePer = Math.max(maxProfilePer, p[0] / p[1]);
			}
		}
	}
	
	var isEdgeOver = false;
	var isEdgeLabelOver = false;

	// Event handler for dragging groups
	var dragHulls = d3.behavior.drag()
		.origin(function(d) { return d; })
	    .on("dragstart", function(d) {
	    	d3.event.sourceEvent.stopPropagation();
			d3.event.sourceEvent.preventDefault();
			forceLayout.stop();
	    })
	    .on("drag", function dragged(d) {
	    		// Shift all group members
	    		var group = graph.groups[d.group];
				for (var i in group.nodes) {
					var node = group.nodes[i];
					node.x += d3.event.dx;
					node.y += d3.event.dy;
					node.px += d3.event.dx;
					node.py += d3.event.dy;
				}
				group.pos[0] += d3.event.dx;
				group.pos[1] += d3.event.dy;
				// Shift all members of sub groups
				for (var k in group.childs) {
					var cgroup = graph.groupsd[group.childs[k]];
					var nodes = cgroup.nodes;
					for (var j in nodes) {
						var node = nodes[j];
						node.x += d3.event.dx;
						node.y += d3.event.dy;
						node.px += d3.event.dx;
						node.py += d3.event.dy;
						cgroup.pos[0] += d3.event.dx;
						cgroup.pos[1] += d3.event.dy;
					}
				}
				updateGraph();
			})
		.on('dragend', function(d) {forceLayout.resume();});
	
	// Draw convex hull surrounding group of nodes
	graph.hulls = convexHulls(graph);
	hulls = pane.selectAll('#hulls').remove();
	hulls = pane.append('g').attr('id', 'hulls')
		.selectAll('path')
		.data(graph.hulls).enter()
		.append('path')
		.attr('class', 'hull')
		.attr('d', drawConvexHull)
		.call(dragHulls);
		
	// Event handler to open/close groups
	hulls.on('dblclick', function(d) {
		var group = graph.groups[d.group];
		group.parent.value.showChilds = !group.parent.value.showChilds;
		if (!group.parent.value.showChilds) {
			for (i in group.childs) {
				var child = graph.groupsd[group.childs[i]];
				child.parent.value.showChilds = false;
			}
		}
		graph = frontEndGraph(dotGraph, graph);
		drawGraph();
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
			graph = frontEndGraph(dotGraph, graph);
			if (!fixOnInit && d.value.showChilds) {
				var n = dotGraph.neighbors(d.id);
				for (i in n) {
					graph.nodesd[n[i]].fixed = false;
				}
			}
			drawGraph();
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
		// Show node details
		if (!isEditNode) {
		   	nodeInfo.transition()        
		        .duration(200)      
		        .style('opacity', .9);
		    nodeInfo
		    	.html(formatNodeInfos(node))
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
	   	hideNodeInfo();
	});
	
	nodes.on('contextmenu', d3.contextMenu(menuItems));
	
	forceLayout = d3.layout.force()
		.nodes(graph.nodes)
		.links(graph.edges)
		.size(graph.size)
		.linkDistance(200)
		.charge(-1000)
		.linkStrength(0.2)
		.gravity(0.05)
		.friction(0.5)
		.on('tick', updateGraph)
		.start();
		
	// Drag behavour
	var drag = forceLayout.drag()
		.on('dragstart', function(d) {
			d3.event.sourceEvent.stopPropagation();
			d3.event.sourceEvent.preventDefault();
			d.fixed = true;
		});
	nodes.call(drag);
}


/*
 * Computes weighted average between two points.
 */
function avgPos(x1, y1, x2, y2, c) {
	x = (1 - c) * x1 + c * x2;
	y = (1 - c) * y1 + c * y2;
	p = x + ',' + y;
	return p;
}


/*
 * Checks for collisions in d3.js quadtree.
 * See http://bl.ocks.org/mbostock/3231298 for more details.
 */
function collide(node) {
	var eps = 10;
	var nx1 = node.x - node.value.cx - eps;
	var nx2 = node.x + node.value.cx + eps;
	var ny1 = node.y - node.value.cy - eps;
	var ny2 = node.y + node.value.cy + eps;
	return function(quad, x1, y1, x2, y2) {
		var point = quad.point;
		if (point && (point != node) && !point.fixed && ! node.fixed) {
			var px1 = point.x - point.value.cx;
			var px2 = point.x + point.value.cx;
			var py1 = point.y - point.value.cy;
			var py2 = point.y + point.value.cy;
			if (!(px1 > nx2 || px2 < nx1 || py1 >= ny2 || py2 <= ny1)) {
				var eta = 0.1;
				if (px1 < nx1) {
					// move quad to left
					var d = eta * (px2 - nx1);
					point.x -= d;
					node.x += d;
				} else {
					var d = eta * (nx2 - px1);
					point.x += d;
					node.x -= d;
				}
				if (py1 < ny1) {
					// move quad to top
					var d = eta * (py2 - ny1);
					point.y -= d;
					node.y += d;
				} else {
					var d = eta * (ny2 - py1);
					point.y += d;
					node.y -= d;
				}
			}
		}
		return x1 > nx2 || x2 < nx1 || y1 >= ny2 || y2 <= ny1;
	};
}


/*
 * Computes euclidean distance between points.
 */
function distance(x1, y1, x2, y2) {
	return Math.sqrt(Math.pow(x1-x2, 2) + Math.pow(y1-y2, 2));
}


/*
 * Updates graph visualization.
 */
function updateGraph() {
	// Avoid collisions
 	var q = d3.geom.quadtree(graph.nodes);
	for (var i in graph.nodes) {
		q.visit(collide(graph.nodes[i]));
	}
		
	graph.hulls = convexHulls(graph);
	hulls.data(graph.hulls)
		.attr('d', drawConvexHull);
	
	// Update nodes
	nodes.attr('transform', function(d) { return 'translate(' + (d.x - d.value.cx) + ' ' + (d.y - d.value.cy) + ')'; });
	// Update edges
	edges.attr('d', function(d) {
		var dist = 100;
		var l = distance(d.source.x, d.source.y, d.target.x, d.target.y);
		var n = Math.max(2, Math.floor(l / dist));
		var marker = [];
		for (var i = 1; i < n; ++i) {
			marker.push(i / n);
		}
		var markerPos = marker.map(function(c) {
			return avgPos(d.source.x, d.source.y, d.target.x, d.target.y, c);});
		var markerPos = ' L' + markerPos.join(' L');
		return 'M' + d.source.x + ',' + d.source.y + markerPos + ' L' + d.target.x + ',' + d.target.y;
	});
}


/*
 * Toggles between usual nodes colors and profiling colors
 */	
function toggleNodeColors() {
		useProfileColors = !useProfileColors;
		updateNodes();
		updateGraph();
	}


/*
 * Computes bounding box that fits text of a certain length.
 */	
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


/*
 * Computes profiling color.
 */
function profileColor(per) {
	var s = d3.scale.linear()
		.domain(linspace(0, maxProfilePer, profileColors.length))
		.range(profileColors)
		.interpolate(d3.interpolateRgb);
	return s(per);
}


/*
 * Returns node fill color.
 */
function nodeFillColor(d) {
	if (useProfileColors) {
		var p = d.value.profile;
		if (d.value.node_type == 'apply' && exists(p)) {
			return profileColor(d.value.profile[0] / d.value.profile[1]);
		} else {
			return 'white';
		}
	} else {
		return typeof(d.value.fillcolor) == 'undefined' ? 'white' : d.value.fillcolor;
	}
}


/*
 * Formats profiling timing information.
 */
function formatTime(sec) {
	var s;
	if (sec < 0.1) {
		s = (sec * 1000).toFixed(1) + ' ms';
	} else {
		s = sec.toFixed(1) + ' s';
	}
	return s;
}


/*
 * Formats node details.
 */
function formatNodeInfos(node) {
	var v = node.value;
	var s = '<b><center>' + v.label + '</center></b><hr>';
	s += '<b>Node:</b> ' + replaceAll(v.node_type, '_', ' ') + ' node';
	if (exists(v.dtype)) {
		s += '</br>';
		s += '<b>Type:</b> <source>' + v.dtype + '</source>';
	}
	if (exists(v.apply_op)) {
		s += '</br>';
		s += '<b>Apply:</b> <source>' + v.apply_op + '</source>';
	}
	if (exists(v.tag)) {
		s += '<p>';
		s += '<b>Location:</b> <source>' + v.tag[1] + ': ' + v.tag[0] + '</source><br>';
		s += '<b>Definition:</b> <source>' + v.tag[2] + '</source><br>';
		s += '</p>';
	}
	var p = v.profile;
	if (exists(p)) {
		s += '<p>';
		s += '<b>Time:</b> ' + formatTime(p[0]);
		s += ' / ' + (p[0] / p[1] * 100).toFixed(1) + ' %';
		s += '</p>';
	}
	return s;	
}


/*
 * Updates node visualization.
 */
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
	shape.attr('fill', nodeFillColor(d));
	
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


/*
 * Updates visualization of all nodes.
 */
function updateNodes() {
	nodes.each(function(d) {
		var node = d3.select(this);
		updateNode(d, node);
	});	
}


/*
 * Hides node information field.
 */
function hideNodeInfo() {
	nodeInfo.transition()        
        .duration(200)      
        .style('opacity', 0);
}


/*
 * Adjusts node size.
 */
function setNodeSize(node) {
	var size = textSize(node.value.label, {'class': 'nodeText'});
		node.value.width = size.width + 2 * pad;
		node.value.height = size.height + 2 * pad;
		node.value.cx = node.value.width / 2;
		node.value.cy = node.value.height / 2;
	}


/* 
 * Event handler for editing nodes.
 */
function editNode(elm, d) {
		var node = d3.select(elm);
		var pos = elm.getBBox();
		if (d3.event.defaultPrevented) return;
		
		isEditNode = true;
		hideNodeInfo();
		
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


/*
 * Release node from fixed positions.
 */
function releaseNode(d) {
	d.fixed = false;
	forceLayout.start();
}


/*
 * Releases positions of all nodes.
 */
function releaseNodes() {
	graph['nodes'].forEach (function (d) {
		d.fixed = false;
	});
	forceLayout.start();
}


/*
 * Restores original node positions.
 */
function resetNodes() {
	graph = frontEndGraph(dotGraph);
	drawGraph();
}
