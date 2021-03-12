#pragma once
#include <string>
#include <fstream>
#include <string>
#include <CGAL/IO/OBJ_reader.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <CGAL/Polygon_mesh_processing/border.h>

namespace sk {
	constexpr size_t NODE_TYPE_END = 0;
	constexpr size_t NODE_TYPE_JUNCT = 1;
	constexpr size_t NODE_TYPE_MID = 2;

	typedef CGAL::Simple_cartesian<double>                        Kernel;
	typedef Kernel::Point_3                                       Point;
	typedef CGAL::Surface_mesh<Point>                             Surface;
	typedef Surface::Vertex_index                                 vertex_descriptor;
	typedef Surface::Face_index                                   face_descriptor;
	typedef boost::graph_traits<Surface>::halfedge_descriptor   halfedge_descriptor;
	typedef CGAL::Polyhedron_3<Kernel>                            Polyhedron;
	typedef CGAL::Mean_curvature_flow_skeletonization<Surface> Skeletonization;
	typedef Skeletonization::Skeleton                             Skeleton;
	typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
	typedef Skeleton::edge_descriptor                             Skeleton_edge;

	bool isSmallHole(halfedge_descriptor h, Surface& mesh,
									 double max_hole_diam, int max_num_hole_edges) {
		int num_hole_edges = 0;
		CGAL::Bbox_3 hole_bbox;
		for (halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh)) {
			const Point& p = mesh.point(target(hc, mesh));
			hole_bbox += p.bbox();
			++num_hole_edges;

			if (num_hole_edges > max_num_hole_edges) return false;
			if (hole_bbox.xmax() - hole_bbox.xmin() > max_hole_diam) return false;
			if (hole_bbox.ymax() - hole_bbox.ymin() > max_hole_diam) return false;
			if (hole_bbox.zmax() - hole_bbox.zmin() > max_hole_diam) return false;
		}
		std::cout << num_hole_edges << std::endl;
		return true;
	}

	struct Node {
		glm::vec3 pos;
		size_t id;
		std::shared_ptr<Node> parent;
		std::vector<std::shared_ptr<Node>> ad_nodes;
		std::vector<size_t> surface_index;

		Node() : pos({ 0, 0, 0 }), id(0), parent(nullptr), ad_nodes({}), surface_index({}) {}
		Node(const glm::vec3 v) : pos(v), id(0), parent(nullptr), ad_nodes({}), surface_index({}) {}
		Node(const glm::vec3 v, const size_t id) : pos(v), id(id), parent(nullptr), ad_nodes({}), surface_index({}) {}

		bool operator==(const Node& other) const {
			return pos == other.pos;
		}

		size_t type() {
			if (this->ad_nodes.size() < 2 && this->parent == nullptr) {
				return NODE_TYPE_END;
			} else if (this->ad_nodes.size() > 2 && this->parent == nullptr) {
				return NODE_TYPE_JUNCT;
			} else {
				return NODE_TYPE_MID;
			}
		}

	};

	struct CurveSkeleton {
		std::vector<std::shared_ptr<Node>> nodes;

		double meanEdgeLength() {
			double sum = 0.0;
			size_t count = 0;
			for (auto& n : nodes) {
				for (auto& ad : n->ad_nodes) {
					sum += glm::length(n->pos - ad->pos);
					count++;
				}
			}
			return sum / count;
		}

		std::pair<std::vector<glm::vec3>, std::vector<glm::vec3>> convertDrawable(
			const glm::vec3 end = { 1.0, 0.0, 0.0 }, const glm::vec3 junct = { 0.0, 1.0, 0.0 }, const glm::vec3 mid = { 0.2, 0.2, 0.2 }) {
			std::vector<glm::vec3> pos, color;
			for (auto& v : nodes) {
				pos.emplace_back(v->pos);

				if (v->ad_nodes.size() < 2 && v->parent == nullptr) color.emplace_back(end);
				else if (v->ad_nodes.size() > 2 && v->parent == nullptr) color.emplace_back(junct);
				else color.emplace_back(mid);
			}

			return { pos, color };
		}

	};

	template <typename T1, typename T2>
	CurveSkeleton createSkeleton(const std::vector<T1>& points, const std::vector<T2>& indices, 
															 const std::string& path = "") {
		std::unordered_map<glm::vec3, size_t> unique_nodes;
		CurveSkeleton cs;

		Surface mesh;
		for (auto& p : points) mesh.add_vertex(Point(p[0], p[1], p[2]));
		for (size_t i = 0; i < indices.size(); i += 3) {
			mesh.add_face(CGAL::SM_Vertex_index(indices[i + 0]),
										CGAL::SM_Vertex_index(indices[i + 1]),
										CGAL::SM_Vertex_index(indices[i + 2]));
		}

		std::vector<halfedge_descriptor> halfedge_descriptors;
		CGAL::Polygon_mesh_processing::extract_boundary_cycles(mesh, std::back_inserter(halfedge_descriptors));
		for (auto& e : halfedge_descriptors) {
			std::vector<face_descriptor>  patch_facets;
			std::vector<vertex_descriptor> patch_vertices;

			bool success = std::get<0>(
				CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(mesh, e,
																																				std::back_inserter(patch_facets), std::back_inserter(patch_vertices)));

		}

		Skeleton skeleton;
		CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

		std::ofstream output;
		if (!path.empty()) output.open(path + "_skeleton.cgal");
		for (size_t i = 0; Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
			const Point& s = skeleton[source(e, skeleton)].point;
			const Point& t = skeleton[target(e, skeleton)].point;

			bool c1 = unique_nodes.contains(glm::vec3(s[0], s[1], s[2]));
			bool c2 = unique_nodes.contains(glm::vec3(t[0], t[1], t[2]));

			size_t si, ti;
			if (c1 && c2) {
				si = unique_nodes[glm::vec3(s[0], s[1], s[2])];
				ti = unique_nodes[glm::vec3(t[0], t[1], t[2])];
			} else if (c1) {
				unique_nodes[glm::vec3(t[0], t[1], t[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(t[0], t[1], t[2]), i)));
				si = unique_nodes[glm::vec3(s[0], s[1], s[2])];
				ti = i++;
			} else if (c2) {
				unique_nodes[glm::vec3(s[0], s[1], s[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(s[0], s[1], s[2]), i)));
				si = i++;
				ti = unique_nodes[glm::vec3(t[0], t[1], t[2])];
			} else {
				unique_nodes[glm::vec3(s[0], s[1], s[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(s[0], s[1], s[2]), i)));
				si = i++;
				unique_nodes[glm::vec3(t[0], t[1], t[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(t[0], t[1], t[2]), i)));
				ti = i++;
			}
			cs.nodes[si]->ad_nodes.emplace_back(cs.nodes[ti]);
			cs.nodes[ti]->ad_nodes.emplace_back(cs.nodes[si]);

			if (!path.empty()) output << "2 " << s << " " << t << "\n";
		}
		if (!path.empty()) output.close();

		if (!path.empty()) output.open(path + "_correspondance_sk.cgal");
		for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
			for (vertex_descriptor vd : skeleton[v].vertices) {
				cs.nodes[unique_nodes[glm::vec3(skeleton[v].point[0], skeleton[v].point[1], skeleton[v].point[2])]]->surface_index.push_back(vd.idx());
				if (!path.empty()) output << "2 " << skeleton[v].point << " " << vd.idx() << "\n";
			}
		}
		if (!path.empty()) output.close();

		return cs;
	}

	CurveSkeleton loadSkeleton(const std::string& path) {
		std::unordered_map<glm::vec3, size_t> unique_nodes;
		CurveSkeleton cs;

		std::ifstream input(path + "_skeleton.cgal");
		std::string line;
		size_t i = 0;
		while (std::getline(input, line)) {
			auto sline = mio::split(line);
			if (sline.size() < 3) continue;

			glm::vec3 s = glm::vec3(std::stod(sline[1]), std::stod(sline[2]), std::stod(sline[3]));
			glm::vec3 t = glm::vec3(std::stod(sline[4]), std::stod(sline[5]), std::stod(sline[6]));

			bool c1 = unique_nodes.contains(glm::vec3(s[0], s[1], s[2]));
			bool c2 = unique_nodes.contains(glm::vec3(t[0], t[1], t[2]));

			size_t si, ti;
			if (c1 && c2) {
				si = unique_nodes[glm::vec3(s[0], s[1], s[2])];
				ti = unique_nodes[glm::vec3(t[0], t[1], t[2])];
			} else if (c1) {
				unique_nodes[glm::vec3(t[0], t[1], t[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(t[0], t[1], t[2]), i)));
				si = unique_nodes[glm::vec3(s[0], s[1], s[2])];
				ti = i++;
			} else if (c2) {
				unique_nodes[glm::vec3(s[0], s[1], s[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(s[0], s[1], s[2]), i)));
				si = i++;
				ti = unique_nodes[glm::vec3(t[0], t[1], t[2])];
			} else {
				unique_nodes[glm::vec3(s[0], s[1], s[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(s[0], s[1], s[2]), i)));
				si = i++;
				unique_nodes[glm::vec3(t[0], t[1], t[2])] = i;
				cs.nodes.emplace_back(std::make_shared<Node>(Node(glm::vec3(t[0], t[1], t[2]), i)));
				ti = i++;
			}
			cs.nodes[si]->ad_nodes.emplace_back(cs.nodes[ti]);
			cs.nodes[ti]->ad_nodes.emplace_back(cs.nodes[si]);
		}
		input.close();

		input.open(path + "_correspondance_sk.cgal");
		while (std::getline(input, line)) {
			auto sline = mio::split(line);
			if (sline.size() < 3) continue;

			glm::vec3 v = glm::vec3(std::stod(sline[1]), std::stod(sline[2]), std::stod(sline[3]));
			cs.nodes[unique_nodes[v]]->surface_index.push_back(std::stoi(sline[4]));
		}
		input.close();

		return cs;
	}

	std::shared_ptr<Node> getRootNode(std::shared_ptr<Node> node) {
		while (node->parent != nullptr) node = node->parent;
		return node;
	}

	void segmentation(CurveSkeleton& skeleton) {
		std::vector<std::shared_ptr<Node>> junctions;

		for (auto& node : skeleton.nodes) {
			if (node->ad_nodes.size() < 2) {
				auto root = node;
				auto from = node;
				auto next = node->ad_nodes[0];
				while (root->id != next->id && next->ad_nodes.size() == 2) {
					next->parent = root;
					auto prev = from;
					from = next;
					next = (next->ad_nodes[0] == prev) ? next->ad_nodes[1] : next->ad_nodes[0];
				}
			} else if (node->ad_nodes.size() > 2) {
				junctions.emplace_back(node);
			}

		}

		for (auto& node : junctions) {
			for (auto next : node->ad_nodes) {
				if (next->parent != nullptr) continue;

				std::vector<std::shared_ptr<Node>> tempPath;
				auto root = node;
				auto from = node;
				while (root->id != next->id && next->ad_nodes.size() == 2) {
					tempPath.emplace_back(next);
					auto prev = from;
					from = next;
					next = (next->ad_nodes[0] == prev) ? next->ad_nodes[1] : next->ad_nodes[0];
				}

				for (size_t i = 0; i < tempPath.size() / 2; i++) {
					tempPath[i]->parent = root;
				}
				root = next;
				for (size_t i = tempPath.size() / 2; i < tempPath.size(); i++) {
					tempPath[i]->parent = root;
				}
			}

		}

	}

	void merge(CurveSkeleton& skeleton, size_t minEdge = 10, double minDistEnd = 10.0, double minDistJunct = 10.0) {
		for (auto& node : skeleton.nodes) {
			if (node->ad_nodes.size() < 2 && node->parent == nullptr) {
				auto root = node;
				std::vector<std::tuple<size_t, std::shared_ptr<Node>, std::shared_ptr<Node>>> queue;
				queue.emplace_back(std::tuple{ 0, node, node->ad_nodes[0] });

				while (queue.size() > 0) {
					auto [c, prev, next] = queue.back();
					queue.pop_back();
					if (c > minEdge) continue;

					if (next->ad_nodes.size() < 2 && (glm::length(root->pos - next->pos) < minDistEnd)) next->parent = root;

					for (auto search : next->ad_nodes) {
						if (search == prev) continue;
						queue.emplace_back(std::tuple{ c + 1, next, search });
					}
				}

			} else if (node->ad_nodes.size() > 2 && node->parent == nullptr) {
				auto root = node;
				for (auto ad_node : node->ad_nodes) {
					std::vector<std::tuple<size_t, std::shared_ptr<Node>, std::shared_ptr<Node>>> queue;
					queue.emplace_back(std::tuple{ 0, node, ad_node });

					while (queue.size() > 0) {
						auto [c, prev, next] = queue.back();
						queue.pop_back();
						if (c > minEdge) continue;

						if (next->ad_nodes.size() > 2 && (glm::length(root->pos - next->pos) < minDistJunct)) next->parent = root;

						for (auto search : next->ad_nodes) {
							if (search == prev) continue;
							queue.emplace_back(std::tuple{ c + 1, next, search });
						}
					}
				}
			}

		}
	}

	void searchCorrespondence(const std::vector<std::shared_ptr<Node>>& src, const std::vector<std::shared_ptr<Node>>& trg,
														const size_t n, double sum, size_t i, size_t used,
														std::vector<size_t> corr, double& min_sum, std::vector<size_t>& ans) {
		if (i == n) {
			if (sum < min_sum) {
				ans = corr;
				min_sum = sum;
			}
			return;
		}

		if (sum > min_sum) return;
		for (size_t t = 0; t < n; t++) {
			if ((used >> t) & 1) continue;
			corr.push_back(t);
			searchCorrespondence(src, trg, n, sum + glm::length(src[i]->pos - trg[t]->pos), i + 1, (used | (size_t)pow(2, t)), corr, min_sum, ans);
			corr.pop_back();
		}
	}

	auto calcCorrespondence(const CurveSkeleton& src, const CurveSkeleton& trg) {
		/* TO DO
		* 		ワールド座標以外の値を参考に対応付けを求める
		*/
		std::vector<std::shared_ptr<Node>> src_end, src_junct, trg_end, trg_junct;

		for (auto& node : src.nodes) {
			if (node->ad_nodes.size() < 2 && node->parent == nullptr) src_end.emplace_back(node);
			if (node->ad_nodes.size() > 2 && node->parent == nullptr) src_junct.emplace_back(node);
		}

		for (auto& node : trg.nodes) {
			if (node->ad_nodes.size() < 2 && node->parent == nullptr) trg_end.emplace_back(node);
			if (node->ad_nodes.size() > 2 && node->parent == nullptr) trg_junct.emplace_back(node);
		}

		std::vector<size_t> end_corr, junct_corr;
		const size_t ne = std::min(src_end.size(), trg_end.size());
		const size_t nj = std::min(src_junct.size(), trg_junct.size());
		double min_sum = std::numeric_limits<double>::max();
		searchCorrespondence(src_end, trg_end, ne, 0, 0, 0,
												 {}, min_sum, end_corr);

		min_sum = std::numeric_limits<double>::max();
		searchCorrespondence(src_junct, trg_junct, nj, 0, 0, 0,
												 {}, min_sum, junct_corr);

		std::unordered_map<size_t, std::vector<size_t>> correspondence;

		for (size_t i = 0; i < ne; i++) {
			correspondence[trg_end[end_corr[i]]->id].push_back(src_end[i]->id);
		}
		for (size_t i = 0; i < nj; i++) {
			correspondence[trg_junct[junct_corr[i]]->id].push_back(src_junct[i]->id);
		}

		return correspondence;
	}

	std::tuple<size_t, std::vector<size_t>, std::vector<size_t>> segmentVector(const size_t src_n, const size_t trg_n,
																																						 const CurveSkeleton& src_skeleton, const CurveSkeleton& trg_skeleton,
																																						 std::unordered_map<size_t, std::vector<size_t>>& correspondence) {
		std::vector<size_t> src_segment(src_n, 0), trg_segment(trg_n, 0);
		std::unordered_map<size_t, size_t> src_group, trg_group;
		size_t id = 1;

		for (auto& v : trg_skeleton.nodes) {
			size_t from = sk::getRootNode(v)->id;
			auto&& temp = correspondence[from];

			if (temp.size() == 0) continue;

			size_t seg_id = 0;
			if (trg_group.contains(from)) {
				seg_id = trg_group[from];
				for (size_t root : temp) {
					src_group[root] = seg_id;
				}
			} else {
				for (size_t root : temp) {
					if (src_group.contains(root)) {
						seg_id = src_group[root];
					}
				}

				if (seg_id == 0) seg_id = id++;

				trg_group[from] = seg_id;
				for (size_t root : temp) {
					src_group[root] = seg_id;
				}
			}

			for (size_t root : temp) {
				for (auto i : v->surface_index) {
					if (i >= trg_n) continue;
					trg_segment[i] = seg_id;
				}
			}
		}

		for (auto& v : src_skeleton.nodes) {
			size_t root = sk::getRootNode(v)->id;
			size_t seg_id = 0;
			if (src_group.contains(root)) { seg_id = src_group[root]; } else continue;

			for (auto i : v->surface_index) {
				if (i >= src_n) continue;
				src_segment[i] = seg_id;
			}
		}

		return { id - 1, src_segment, trg_segment };
	}

}