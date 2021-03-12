#define NOMINMAX
#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING

#include <iostream>
#include <string>
#include <numbers>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "Eigen/Core"
#include "Eigen/Sparse"

#include "glTFio.hpp"
#define IMGUI_IMPLEMENTATION
#define FRAME_COMMAND
#include "simple_sketch.hpp"
#include "meshio.hpp"
#include "halfedge.hpp"
#include "geoutil.hpp"
#include "geoproc.hpp"
#include "laplace.hpp"
#include "eigen_rapper.hpp"
#include "feature.hpp"

#include "skeleton.hpp"
#include "functional_map.hpp"

constexpr size_t NUMEIGS = 50;
using Evec = Eigen::Matrix<double, NUMEIGS, 1>;

struct ExGuiParam {
	bool disp_segment{};
	bool disp_descriptor{};
	bool stop_refine{};
	bool recreate{};
} GUIPARAM;

void guiAddParam() {
	ImGui::Checkbox("Segment",     &GUIPARAM.disp_segment);
	ImGui::Checkbox("Descriptor",  &GUIPARAM.disp_descriptor);
	ImGui::Checkbox("StopRefine",  &GUIPARAM.stop_refine);
	ImGui::Checkbox("ComputeFMap", &GUIPARAM.recreate);
}

struct SketchData {
	bool signal{};
	bool destroy{};

	size_t num_segment;
	std::vector<size_t> src_segment;
	std::vector<size_t> trg_segment;

	size_t hks_samples, wks_samples;
	size_t num_mapped, num_samples;
	Eigen::MatrixXd src_hks_block, src_wks_block;
	Eigen::MatrixXd trg_hks_block, trg_wks_block;

	FuncMap<Evec> map;
	std::vector<int> correspondence;
	std::thread th_refine;

	std::pair<size_t, size_t> select{ std::numeric_limits<size_t>::max(), 
																		std::numeric_limits<size_t>::max() };
	sk::CurveSkeleton src_skeleton, trg_skeleton;
	std::unordered_map<size_t, std::vector<size_t>> sk_corr;

};

void SimpSketch<SketchData>::frameCommand() {
	bool update = false;

	if (var->signal) {
		std::cout << "REFINE" << std::endl;
		var->signal = false;
		update = true;
		updateVertexBuffer();
	}

	if (GUIPARAM.stop_refine) {
		update = true;
	}

	if (eventManager.click.action == GLFW_PRESS &&
			eventManager.click.button == GLFW_MOUSE_BUTTON_2) eventManager.actionSignal = 1;

	if (var->select.first != std::numeric_limits<size_t>::max() &&
			var->select.second != std::numeric_limits<size_t>::max() &&
			eventManager.keyboard.action == GLFW_PRESS) {
		if (eventManager.keyboard.key == GLFW_KEY_A) eventManager.actionSignal = 2;
		if (eventManager.keyboard.key == GLFW_KEY_S) eventManager.actionSignal = 3;
		if (eventManager.keyboard.key == GLFW_KEY_D) eventManager.actionSignal = 4;
	}

	if (eventManager.click.action == GLFW_RELEASE && eventManager.actionSignal == 1) {
		eventManager.actionSignal = 0;
		glm::vec2 p = glm::vec2(eventManager.pos.x / swapChain.extent.width, eventManager.pos.y / swapChain.extent.height);
		auto [sk_nodes0, sk_color0] = var->src_skeleton.convertDrawable();
		auto [sk_nodes1, sk_color1] = var->trg_skeleton.convertDrawable();

		int i = 0;
		int min_index = -1;
		double min_dist = std::numeric_limits<double>::max();
		for (auto& node : var->src_skeleton.nodes) {
			if (node->parent == nullptr) {
				auto v = glm::vec4(node->pos, 1.0);
				auto tv = ubo.proj * ubo.view * models[0].matrix * v;
				auto n = glm::vec2(tv / tv.w) * 0.5f + 0.5f;

				if (glm::length(p - n) < min_dist) {
					min_dist = glm::length(p - n);
					min_index = i;
				}
			}
			i++;
		}

		for (auto& node : var->trg_skeleton.nodes) {
			if (node->parent == nullptr) {
				auto v = glm::vec4(node->pos, 1.0);
				auto tv = ubo.proj * ubo.view * models[1].matrix * v;
				auto n = glm::vec2(tv / tv.w) * 0.5f + 0.5f;

				if (glm::length(p - n) < min_dist) {
					min_dist = glm::length(p - n);
					min_index = i;
				}
			}
			i++;
		}

		if (min_index >= 0) {
			drawPoints[min_index].color = { 0.0, 0.0, 1.0 };

			if (min_index >= sk_nodes0.size()) var->select.second = min_index - sk_nodes0.size();
			else var->select.first = min_index;
		}
		updateMiscBuffer();
	}

	if (eventManager.keyboard.action == GLFW_RELEASE &&
			eventManager.actionSignal >= 2 && eventManager.actionSignal <= 4) {

		auto f = var->select.first;
		auto s = var->select.second;

		if (eventManager.actionSignal == 2) var->sk_corr[s].push_back(f);
		else if (eventManager.actionSignal == 3) var->sk_corr[s] = { f };
		else if (eventManager.actionSignal == 4) {
			var->sk_corr[s].clear();
			var->sk_corr[s].shrink_to_fit();
		}

		std::tie(var->num_segment, var->src_segment, var->trg_segment) = sk::segmentVector(models[0].vertices.size(), models[1].vertices.size(),
																																											 var->src_skeleton, var->trg_skeleton, 
																																											 var->sk_corr);

		for (size_t i = 0; i < var->src_segment.size(); i++) {
			if (var->src_segment[i] == 0) models[0].vertices[i].color = glm::vec3(0.0);
			else models[0].vertices[i].color = gp::color::randomColor(var->src_segment[i]);
		}
		for (size_t i = 0; i < var->trg_segment.size(); i++) {
			if (var->trg_segment[i] == 0) models[1].vertices[i].color = glm::vec3(0.0);
			else models[1].vertices[i].color = gp::color::randomColor(var->trg_segment[i]);
		}

		drawPoints[f].color = (var->src_skeleton.nodes[f]->type() == sk::NODE_TYPE_END) ? glm::vec3{ 1.0, 0.0, 0.0 } : glm::vec3{ 0.0, 1.0, 0.0 };
		drawPoints[s + var->src_skeleton.nodes.size()].color =
			(var->trg_skeleton.nodes[s]->type() == sk::NODE_TYPE_END) ? glm::vec3{ 1.0, 0.0, 0.0 } : glm::vec3{ 0.0, 1.0, 0.0 };

		drawLines.clear();
		drawLines.shrink_to_fit();
		for (auto& kv : var->sk_corr) {
			for (auto& v : kv.second) {
				Line l{};
				l.s.pos = models[0].matrix * glm::vec4(var->src_skeleton.nodes[v]->pos, 1.0);
				l.t.pos = models[1].matrix * glm::vec4(var->trg_skeleton.nodes[kv.first]->pos, 1.0);
				drawLines.push_back(l);
			}
		}

		updateVertexBuffer();
		createMiscBuffer();
		var->select = { std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() };

		eventManager.actionSignal = 0;

	}

	if (GUIPARAM.stop_refine && GUIPARAM.recreate) {
		GUIPARAM.recreate = false;
		var->destroy = true;

		if (var->th_refine.joinable()) var->th_refine.join();

		auto m0n = models[0].vertices.size(), m1n = models[1].vertices.size();
		std::cout << "Recreate" << std::endl;
		for (size_t i = 1; i < var->num_segment + 1; i++) {
			Eigen::VectorXd src_mask = Eigen::VectorXd::Zero(m0n);
			Eigen::VectorXd trg_mask = Eigen::VectorXd::Zero(m1n);
			for (size_t t = 0; t < m0n; t++) src_mask(t) = (var->src_segment[t] == i);
			for (size_t t = 0; t < m1n; t++) trg_mask(t) = (var->trg_segment[t] == i);

			var->map.src_desc.block(0, 0 + (i - 1) * var->num_mapped, m0n, var->hks_samples) =
				src_mask.asDiagonal() * var->src_hks_block;
			var->map.src_desc.block(0, var->hks_samples + (i - 1) * var->num_mapped, m0n, var->wks_samples) =
				src_mask.asDiagonal() * var->src_wks_block;
			var->map.trg_desc.block(0, 0 + (i - 1) * var->num_mapped, m1n, var->hks_samples) =
				trg_mask.asDiagonal() * var->trg_hks_block;
			var->map.trg_desc.block(0, var->hks_samples + (i - 1) * var->num_mapped, m1n, var->wks_samples) =
				trg_mask.asDiagonal() * var->trg_wks_block;
		}

		var->map.src_desc.normalize();
		var->map.trg_desc.normalize();

		std::fill(var->correspondence.begin(), var->correspondence.end(), 0);
		size_t num_refine = 50;

		var->destroy = false;
		var->th_refine = std::move(std::thread(&FuncMap<Eigen::Matrix<double, NUMEIGS, 1>>::refineMT,
																							 &var->map, num_refine, std::ref(var->correspondence),
																							 std::ref(var->signal), std::ref(GUIPARAM.stop_refine), std::ref(var->destroy)));
	}

	if (update) {
		if (GUIPARAM.disp_segment) {
			for (size_t i = 0; i < models[0].vertices.size(); i++) {
				if (var->src_segment[i] == 0) models[0].vertices[i].color = glm::vec3(0.0);
				else models[0].vertices[i].color = gp::color::randomColor(var->src_segment[i]);
			}

			for (size_t i = 0; i < models[1].vertices.size(); i++) {
				if (var->trg_segment[i] == 0) models[1].vertices[i].color = glm::vec3(0.0);
				else models[1].vertices[i].color = gp::color::randomColor(var->trg_segment[i]);
				models[1].vertices[i].texCoord = models[0].vertices[var->correspondence[i]].texCoord;
			}
		} else if (GUIPARAM.disp_descriptor) {
			size_t fc = (size_t)(frameTime * 8.0) % (var->num_samples);
			double min = var->map.src_desc.col(fc).minCoeff(), max = var->map.src_desc.col(fc).maxCoeff();
			for (size_t i = 0; i < models[0].vertices.size(); i++) {
				models[0].vertices[i].color = gp::color::scalar2rgb<glm::vec3>(gp::mapping(var->map.src_desc(i, fc), min, max));
			}
			min = var->map.trg_desc.col(fc).minCoeff(), max = var->map.trg_desc.col(fc).maxCoeff();
			for (size_t i = 0; i < models[1].vertices.size(); i++) {
				models[1].vertices[i].color = gp::color::scalar2rgb<glm::vec3>(gp::mapping(var->map.trg_desc(i, fc), min, max));
			}
		} else {
			for (size_t i = 0; i < models[0].vertices.size(); i++)
				models[0].vertices[i].color = gp::color::scalar2rgb<glm::vec3>((float)i / models[0].vertices.size());
			for (size_t i = 0; i < models[1].vertices.size(); i++) {
				models[1].vertices[i].color = models[0].vertices[var->correspondence[i]].color;
				models[1].vertices[i].texCoord = models[0].vertices[var->correspondence[i]].texCoord;
			}
		}
		updateVertexBuffer();
	}

}

struct Model {
	std::string name{};
	size_t n{};
	std::vector<uint32_t> indices{};
	std::vector<Eigen::Vector3d> pos{};
	std::vector<Eigen::Vector3d> normal{};
	std::vector<Eigen::Vector2d> texcoord{};
	std::vector<glm::vec3> color{};
	std::vector<glm::vec4> joints{};
	std::vector<glm::vec4> weights{};

	heds::Mesh mesh;
	Eigen::SparseMatrix<double> lbo, areas;

	void load(std::string_view path) {
		auto [vert, faces] = mio::loadOBJvf<Eigen::Vector3d>(path.data());
		n = vert.size();
		pos = std::move(vert);
		indices = mio::faces2indices(faces);
		color.resize(n, { 0.8, 0.8, 0.8 });
		normal = gp::calcVertexNormal(pos, indices);
		texcoord = gp::calcVertexTexcoord<Eigen::Vector2d>(pos);
		joints.resize(n, glm::vec4(0.0));
		weights.resize(n, glm::vec4(0.0));

		mesh = std::move(heds::Mesh(pos, faces));
		lbo = std::move(gp::cotanLaplacianSparseMat(mesh, false));
		areas = std::move(gp::areaSparseMat(mesh, 3.0));
	}
};

int main(int argc, char* argv[]) {
	const std::string model_path = "./data/model/";
	const std::string eigen_path = "./data/eigen/";
	const std::string skeleton_path = "./data/skeleton/";
	const std::string fmap_path = "./data/fmap/";

	Model src, trg;
	src.name = std::string(argv[1]);
	src.name = src.name.substr(0, src.name.find("."));
	trg.name = std::string(argv[2]);
	trg.name = trg.name.substr(0, trg.name.find("."));
	
	src.load(model_path + argv[1]);
	trg.load(model_path + argv[2]);

	//---------------------parameter-----------------------
	SketchData sd;

	std::pair<double, double> merge_range = { 10.0, 14.0 };
	sd.hks_samples = 0;
	sd.wks_samples = 100;
	sd.num_mapped = sd.hks_samples + sd.wks_samples;
	double lambda = 20.0;
	size_t num_refine = 30;

	sd.map = FuncMap<Evec>(NUMEIGS, lambda, src.name, trg.name, src.areas, trg.areas);
	
	int load_sk = 0, load_eig = 0, load_v2v = 0;
	for (int i = 3; i < argc; i++) {
		if (!std::strcmp(argv[i], "-s")) {
			if (!std::strcmp(argv[i + 1], "st")) load_sk = 1;
			if (!std::strcmp(argv[i + 1], "s")) load_sk = 2;
			if (!std::strcmp(argv[i + 1], "t")) load_sk = 3;
		} else if (!std::strcmp(argv[i], "-e")) {
			if (!std::strcmp(argv[i + 1], "st")) load_eig = 1;
			if (!std::strcmp(argv[i + 1], "s")) load_eig = 2;
			if (!std::strcmp(argv[i + 1], "t")) load_eig = 3;
		} else if (!std::strcmp(argv[i], "-v")) {
			if (!std::strcmp(argv[i+1], "icp")) load_v2v = 1;
			if (!std::strcmp(argv[i+1], "rhm")) load_v2v = 2;
		}
	}

	//------------------curve skeletons-----------------------
	if (load_v2v == 0) {
		if (load_sk == 1) {
			sd.src_skeleton = std::move(sk::createSkeleton(src.pos, src.indices, skeleton_path + src.name));
			sd.trg_skeleton = std::move(sk::createSkeleton(trg.pos, trg.indices, skeleton_path + trg.name));
		} else if (load_sk == 2) {
			sd.src_skeleton = std::move(sk::createSkeleton(src.pos, src.indices, skeleton_path + src.name));
			sd.trg_skeleton = std::move(sk::loadSkeleton(skeleton_path + trg.name));
		} else if (load_sk == 3) {
			sd.src_skeleton = std::move(sk::loadSkeleton(skeleton_path + src.name));
			sd.trg_skeleton = std::move(sk::createSkeleton(trg.pos, trg.indices, skeleton_path + trg.name));
		} else {
			sd.src_skeleton = std::move(sk::loadSkeleton(skeleton_path + src.name));
			sd.trg_skeleton = std::move(sk::loadSkeleton(skeleton_path + trg.name));
		}

		auto src_sk_mel = sd.src_skeleton.meanEdgeLength(), trg_sk_mel = sd.trg_skeleton.meanEdgeLength();
		sk::segmentation(sd.src_skeleton);
		sk::segmentation(sd.trg_skeleton);
		sk::merge(sd.src_skeleton, sd.src_skeleton.nodes.size() * 0.1, src_sk_mel * merge_range.first, src_sk_mel * merge_range.second);
		sk::merge(sd.trg_skeleton, sd.trg_skeleton.nodes.size() * 0.1, trg_sk_mel * merge_range.first, trg_sk_mel * merge_range.second);
		sd.sk_corr = std::move(sk::calcCorrespondence(sd.src_skeleton, sd.trg_skeleton));

		std::tie(sd.num_segment, sd.src_segment, sd.trg_segment) = sk::segmentVector(src.n, trg.n, sd.src_skeleton, sd.trg_skeleton, sd.sk_corr);
		for (size_t i = 0; i < sd.src_segment.size(); i++) {
			if (sd.src_segment[i] == 0) continue; src.color[i] = gp::color::randomColor(sd.src_segment[i]);
		}
		for (size_t i = 0; i < sd.trg_segment.size(); i++) {
			if (sd.trg_segment[i] == 0) continue; trg.color[i] = gp::color::randomColor(sd.trg_segment[i]);
		}
	}

	//---------------------eigen-----------------------
	if (load_v2v == 0) {
		if (load_eig == 0) {
			std::tie(sd.map.src_evals, sd.map.src_evecs) = gp::loadEigen(src.n, NUMEIGS, eigen_path + src.name);
			std::tie(sd.map.trg_evals, sd.map.trg_evecs) = gp::loadEigen(trg.n, NUMEIGS, eigen_path + trg.name);
		} else {
			std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
			if (load_eig == 3) {
				std::tie(sd.map.src_evals, sd.map.src_evecs) = gp::loadEigen(src.n, NUMEIGS, eigen_path + src.name);
			} else {
				std::tie(sd.map.src_evals, sd.map.src_evecs) = gp::calcMatlabGenEigs(matlabPtr, src.lbo, src.areas, NUMEIGS);
				gp::saveEigen(sd.map.src_evals, sd.map.src_evecs, eigen_path + sd.map.src_name);
			}
			if (load_eig == 2) {
				std::tie(sd.map.trg_evals, sd.map.trg_evecs) = gp::loadEigen(trg.n, NUMEIGS, eigen_path + trg.name);
			} else {
				std::tie(sd.map.trg_evals, sd.map.trg_evecs) = gp::calcMatlabGenEigs(matlabPtr, trg.lbo, trg.areas, NUMEIGS);
				gp::saveEigen(sd.map.trg_evals, sd.map.trg_evecs, eigen_path + sd.map.trg_name);
			}
		}
	}

	//---------------------descriptors------------------
	sd.num_samples = sd.num_mapped * sd.num_segment;
	if (load_v2v == 0) {
		sd.map.src_desc = Eigen::MatrixXd::Zero(src.n, sd.num_samples);
		sd.map.trg_desc = Eigen::MatrixXd::Zero(trg.n, sd.num_samples);

		sd.src_hks_block = std::move(gp::timeStepHKS(sd.map.src_evals, sd.map.src_evecs, sd.hks_samples));
		sd.src_wks_block = std::move(gp::timeStepWKS(sd.map.src_evals, sd.map.src_evecs, sd.wks_samples));
		sd.trg_hks_block = std::move(gp::timeStepHKS(sd.map.trg_evals, sd.map.trg_evecs, sd.hks_samples));
		sd.trg_wks_block = std::move(gp::timeStepWKS(sd.map.trg_evals, sd.map.trg_evecs, sd.wks_samples));

		for (size_t i = 1; i < sd.num_segment + 1; i++) {
			Eigen::VectorXd src_mask = Eigen::VectorXd::Zero(src.n);
			Eigen::VectorXd trg_mask = Eigen::VectorXd::Zero(trg.n);
			for (size_t t = 0; t < src.n; t++) src_mask(t) = (sd.src_segment[t] == i);
			for (size_t t = 0; t < trg.n; t++) trg_mask(t) = (sd.trg_segment[t] == i);

			sd.map.src_desc.block(0, 0 + (i - 1) * sd.num_mapped, src.n, sd.hks_samples) = src_mask.asDiagonal() * sd.src_hks_block;
			sd.map.src_desc.block(0, sd.hks_samples + (i - 1) * sd.num_mapped, src.n, sd.wks_samples) = src_mask.asDiagonal() * sd.src_wks_block;
			sd.map.trg_desc.block(0, 0 + (i - 1) * sd.num_mapped, trg.n, sd.hks_samples) = trg_mask.asDiagonal() * sd.trg_hks_block;
			sd.map.trg_desc.block(0, sd.hks_samples + (i - 1) * sd.num_mapped, trg.n, sd.wks_samples) = trg_mask.asDiagonal() * sd.trg_wks_block;
		}

		sd.map.src_desc.normalize();
		sd.map.trg_desc.normalize();
	}

	//----------------------correspondence------------------
	sd.correspondence.resize(trg.n, 0);
	if (load_v2v == 0) {
		GUIPARAM.stop_refine = true;
		sd.th_refine = std::move(std::thread(&FuncMap<Evec>::refineMT, &sd.map,
																				num_refine, std::ref(sd.correspondence), 
																				std::ref(sd.signal), std::ref(GUIPARAM.stop_refine), std::ref(sd.destroy)));
	} else if (load_v2v == 1) {
		std::ifstream file("./data/v2v/" + src.name + "_to_" + trg.name + ".txt");
		for (size_t i = 0; i < trg.n; i++) {
			std::string line;
			std::getline(file, line);
			sd.correspondence[i] = static_cast<size_t>(std::stod(line)) - 1;
		}

		for (size_t i = 0; i < src.n; i++) src.color[i] = gp::color::scalar2rgb<glm::vec3>((float)i / src.n);
		for (size_t i = 0; i < trg.n; i++) {
			trg.color[i] = src.color[sd.correspondence[i]];
			trg.texcoord[i] = src.texcoord[sd.correspondence[i]];
			trg.joints[i] = src.joints[sd.correspondence[i]];
			trg.weights[i] = src.weights[sd.correspondence[i]];
		}

	} else if (load_v2v == 2) {
		std::ifstream file("./data/v2v/" + src.name + "_to_" + trg.name + "_rhm.txt");
		for (size_t i = 0; i < trg.n; i++) {
			std::string line;
			std::getline(file, line);
			sd.correspondence[i] = static_cast<size_t>(std::stod(line)) - 1;
		}

		for (size_t i = 0; i < src.n; i++) src.color[i] = gp::color::scalar2rgb<glm::vec3>((float)i / src.n);
		for (size_t i = 0; i < trg.n; i++) {
			trg.color[i] = { 0.0, 0.0, 0.0 };
			trg.texcoord[i] = { 0.0, 0.0 };
			for (size_t t = sd.correspondence[i] * 3; t < sd.correspondence[i] * 3 + 3; t++) {
				trg.color[i] +=    src.color[src.indices[t]];
				trg.texcoord[i] += src.texcoord[src.indices[t]];
				trg.joints[i] =    src.joints[src.indices[t]];
				trg.weights[i] +=  src.weights[src.indices[t]];
			}
			trg.color[i] /= 3.0f;
			trg.texcoord[i] /= 3.0f;
			trg.weights[i] /= 3.0f;
		}
	}

	SimpSketch<SketchData> sketch;
	sketch.addModel(src.pos, src.indices, src.color, src.normal, src.texcoord);
	sketch.addModel(trg.pos, trg.indices, trg.color, trg.normal, trg.texcoord);
	sketch.addTextureFromFile("./data/texture.png");
	sketch.translateModel(0, { -0.3, 0.0, 0.0 });
	sketch.translateModel(1, { 0.3, 0.0, 0.0 });
	sketch.scaleModel();

	auto [src_sk_nodes, src_sk_color] = sd.src_skeleton.convertDrawable();
	auto [trg_sk_nodes, trg_sk_color] = sd.trg_skeleton.convertDrawable();
	auto& src_mat = sketch.models[0].matrix;
	auto& trg_mat = sketch.models[1].matrix;
	for (auto& v : src_sk_nodes) v = src_mat * glm::vec4(v, 1.0f);
	for (auto& v : trg_sk_nodes) v = trg_mat * glm::vec4(v, 1.0f);
	sketch.addPoints(src_sk_nodes, src_sk_color);
	sketch.addPoints(trg_sk_nodes, trg_sk_color);

	std::vector<std::pair<glm::vec3, glm::vec3>> lines;
	for (auto& kv : sd.sk_corr) {
		for (auto& v : kv.second) {
			lines.emplace_back(trg_sk_nodes[kv.first], src_sk_nodes[v]);
		}
	}
	sketch.addLines(lines);

	sketch.var = &sd;
	sketch.guiAddParameters = guiAddParam;
	
	sketch.run();

	GUIPARAM.stop_refine = true;
	sd.destroy = true;
	if (sd.th_refine.joinable()) sd.th_refine.join();
	if (load_v2v == 0) sd.map.save(fmap_path);

	return 0;
}