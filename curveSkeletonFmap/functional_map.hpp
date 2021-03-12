#pragma once
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "knn.hpp"
#include "icp.hpp"

template <typename EVEC>
class FuncMap {
public:
	size_t numeigs;
	double lambda;

	std::string src_name, trg_name;
	Eigen::VectorXd src_evals, trg_evals;
	Eigen::MatrixXd src_evecs, trg_evecs;

	Eigen::MatrixXd src_desc, trg_desc;
	Eigen::SparseMatrix<double> src_areas, trg_areas;

	Eigen::MatrixXd data;

	FuncMap() {}
	FuncMap(const size_t numeigs, const double lambda,
			 const std::string src_name, const std::string trg_name,
			 const Eigen::SparseMatrix<double> src_areas,
			 const Eigen::SparseMatrix<double> trg_areas) : numeigs(numeigs), lambda(lambda), src_name(src_name), trg_name(trg_name),
		src_areas(src_areas), trg_areas(trg_areas) {}

	void show() {
		std::ofstream temp("matrix.temp");
		for (size_t i = 0; i < numeigs; i++) {
			for (size_t t = 0; t < numeigs - 1; t++) {
				temp << data(i, t) << " ";
			}
			temp << data(i, numeigs - 1) << "\n";
		}
		temp.close();

		system(("python show_fmap.py matrix.temp " + std::to_string(numeigs)).c_str());
		std::filesystem::remove("matrix.temp");
	}

	void compute() {
		data = Eigen::MatrixXd::Zero(numeigs, numeigs);
		Eigen::MatrixXd src_feat = src_evecs.transpose() * src_areas * src_desc;
		Eigen::MatrixXd trg_feat = trg_evecs.transpose() * trg_areas * trg_desc;

		Eigen::MatrixXd A_fixed = src_feat * src_feat.transpose();
		Eigen::MatrixXd B = src_feat * trg_feat.transpose();

		for (size_t i = 0; i < numeigs; i++) {
			Eigen::VectorXd R = Eigen::pow(src_evals.array() - trg_evals[i], 2);
			Eigen::MatrixXd A = (lambda * R.asDiagonal().toDenseMatrix() + A_fixed).inverse();

			data.row(i) = A * B.col(i);
		}
	}

	void save(const std::string& path) {
		std::ofstream output(path + src_name + "_" + trg_name + "_map.txt");
		for (size_t i = 0; i < data.cols(); i++) {
			for (size_t t = 0; t < data.rows() - 1; t++) {
				output << data(i, t) << " ";
			}
			output << data(i, data.rows() - 1) << "\n";
		}
		output.close();
	}

	void refineMT(const size_t num_iter, std::vector<int>& correspondence, bool& signal, bool& wait, bool& destroy) {
		compute();

		Eigen::MatrixXd src = src_evecs.transpose();
		Eigen::MatrixXd trg = (trg_evecs * data).transpose();
		std::vector<EVEC> src_vecs(src.cols()), trg_vecs(trg.cols());

		for (size_t i = 0; i < num_iter; i++) {
			memcpy(src_vecs[0].data(), src.data(), sizeof(double) * src.size());
			memcpy(trg_vecs[0].data(), trg.data(), sizeof(double) * trg.size());

			correspondence = gp::nearestNeighbors(src_vecs, trg_vecs, 8, src.rows());

			signal = true;
			while (wait) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); if (destroy) return; }

			auto [U, V] = gp::rigidTransformSVD(src_evecs.transpose(), trg_evecs.transpose(), correspondence, false);
			data = U * V.transpose();
			trg = (trg_evecs * data).transpose();
		}

		wait = true;
	}
};
