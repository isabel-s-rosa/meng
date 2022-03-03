#include <cmath>
#include <iostream>
#include <array>
#include <vector>
#include <torch/script.h>
#include <cleri/cleri.h>
#include "extxyz/libextxyz/extxyz_kv_grammar.h"
#include "extxyz/libextxyz/extxyz_kv_grammar.c"
#include "extxyz/libextxyz/extxyz.h"

////////////////////////////////////////////////////////////////////////////////

namespace {

typedef std::array<double, 3> Pos;

//----------------------------------------------------------------------------//

struct Node {
    Node(Pos pos)
    : m_pos(pos)
    {}

    Pos pos() const
    {
        return m_pos;
    }

private:
    Pos m_pos;
};

//----------------------------------------------------------------------------//

struct AtomGraph {
    AtomGraph(torch::Tensor nodes, int num_nodes, double r_cutoff)
    : m_nodes(nodes), m_num_nodes(num_nodes), m_r_cutoff(r_cutoff)
    {
        // int num_edges = 0;
        // for (int i = 0; i < m_num_nodes; i++) {
        //     V.push_back(num_edges);
        //     for (int j = 0; j < m_num_nodes; j++) {
        //         if ((dist_squared(m_nodes[i], m_nodes[j]) <= m_r_cutoff * m_r_cutoff) && i != j) {
        //             E.push_back(j);
        //             num_edges++;
        //         }
        //     }
        // }
        // V.push_back(num_edges);
	int num_edges = 0;
	std::vector<int> srcs;
	std::vector<int> targets;
        for (int i=0; i<m_num_nodes; i++) {
            for (int j=0; j<m_num_nodes; j++) {
		double dist_sq = 0;
                for (int k=0; k<3; k++) {
		    dist_sq += (m_nodes[i][k].item<double>() - m_nodes[j][k].item<double>()) * (m_nodes[i][k].item<double>() - m_nodes[j][k].item<double>());
                }
		if (dist_sq <= r_cutoff * r_cutoff && i != j) {
			// std::cout << "added edge (" << i << ", " << j << ")" << std::endl;
		    srcs.push_back(i);
		    targets.push_back(j);
		    num_edges++;
		}
            }
        }
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	int* srcs_ptr = &srcs[0];
	int* targets_ptr = &targets[0];
        torch::Tensor t1 = torch::from_blob(srcs_ptr, {num_edges}, options=options).to(torch::kInt64);
        torch::Tensor t2 = torch::from_blob(targets_ptr, {num_edges}, options=options).to(torch::kInt64);
	edges_list = torch::stack({t1, t2});
    }

    double r_cutoff() const
    {
        return m_r_cutoff;
    }

    int num_nodes() const
    {
        return m_num_nodes;
    }

    torch::Tensor nodes() const
    {
        return m_nodes;
    }

    std::vector<int> vertices() const
    {
        return V;
    }

    torch::Tensor edges() const
    {
        return edges_list;
    }

private:
    torch::Tensor m_nodes;
    int m_num_nodes;
    double m_r_cutoff;
    std::vector<int> V;
    std::vector<int> E;
    torch::Tensor edges_list;

    double dist_squared(Node n1, Node n2) const
    {
        Pos pos1 = n1.pos();
        Pos pos2 = n2.pos();
        return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) +
               (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) +
               (pos1[2] - pos2[2]) * (pos1[2] - pos2[2]);
    }
};

//----------------------------------------------------------------------------//

} // namespace

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    torch::jit::script::Module module;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return -1;
    }

    FILE *fp = fopen(argv[2], "r");

    cleri_grammar_t *kv_grammar = compile_extxyz_kv_grammar();
    int nat;
    DictEntry *info, *arrays;

    int success = extxyz_read_ll(kv_grammar, fp, &nat, &info, &arrays);
    std::cout << "parsed success " << success << std::endl;
    std::cout << "nat " << nat << std::endl;
    std::cout << "info" << std::endl;
    print_dict(info);
    std::cout << "arrays" << std::endl;
    print_dict(arrays);

    std::vector<torch::jit::IValue> inputs;
    torch::Dict<std::string, torch::Tensor> dictionary;

    for (DictEntry *entry = arrays; entry; entry = entry->next) {
        int rows = 1;
        int cols = 1;
        if (entry->nrows != 0) {
            rows = entry->nrows;
        }
        if (entry->ncols != 0) {
            cols = entry->ncols;
        }
        // std::cout << entry->key << std::endl;
	std::string str1 (entry->key);
	std::string str2 ("pos");
	// std::string str3 ("forces");
        if (entry->data_t == data_f && str1.compare(str2) == 0) {
            torch::Tensor t = torch::from_blob((double*) entry->data, {rows, cols});
	    if (str1.compare(str2) == 0) {
                AtomGraph system(t, rows, 5.0);
		torch::Tensor edges = system.edges();
		dictionary.insert("edge_index", edges);
		std::cout << "Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
	    }
            dictionary.insert(entry->key, t);
       } else if (entry->data_t == data_s) {
            std::cout << (char**) (entry->data) << std::endl;
            int arr[cols];
            for (int j=0; j<cols; j++) {
                std::string type(((char**) (entry->data))[j]);
                if (type == "Hf") {
                    arr[j] = 0;
                } else if (type == "O") {
                    arr[j] = 1;
                }
            }
	    auto options = torch::TensorOptions().dtype(torch::kInt32);
            torch::Tensor t = torch::from_blob(arr, {cols, 1}, options=options).to(torch::kInt64);
            dictionary.insert("atom_types", t);
        }
    }

    torch::Tensor t3 = torch::randint(0, 1, {nat}).to(torch::kLong);
    dictionary.insert("batch", t3);
    inputs.push_back(dictionary);

    // Execute the model and turn its output into a tensor.
    // module.eval();
    std::cout << "before forward" << std::endl;
    auto output = module(inputs).toGenericDict();
    std::cout << "after forward" << std::endl;

    std::cout << "keys\n--------------------------------------------------------" << std::endl;
    for (auto it = output.begin(); it!=output.end(); it++) {
        std::cout << it->key() << std::endl;
    }
    std::cout << "keysdone\n--------------------------------------------------------" << std::endl;
    // size: (1857, 3)
    torch::Tensor forces_tensor = output.at("forces").toTensor().detach();
    std::cout << "FORCES" << std::endl;
    for (int i = 0; i < nat; i++) {
        std::cout << i << ": " << forces_tensor[i][0] << ", " << forces_tensor[i][1] << ", " << forces_tensor[i][2] << std::endl;
    }
    torch::Tensor total_energy_tensor = output.at("total_energy").toTensor();
    torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();
    std::cout << "energy dim: " << atomic_energy_tensor.dim() << std::endl;
    std::cout << "energy dim: " << atomic_energy_tensor.size(0) << ", " << atomic_energy_tensor.size(1) << std::endl;
    std::cout << "ENERGIES" << std::endl;
    for (int i = 0; i < nat; i++) {
        std::cout << i << ": " << atomic_energy_tensor[i][0] << std::endl;
    }
    // auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
    float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];
    std::cout << "Atomic energy sum: " << atomic_energy_sum << std::endl;

    torch::Tensor new_pos = output.at("pos").toTensor().detach();

    // physics stuff
    float temp = 300.0;
    float nvt_q = 334.0;
    float dt = 0.5 * pow(10, -15); //femtoseconds?
    float dtdt = pow(dt, 2);
    float nvt_bath = 0.0;
    torch::Tensor atom_types = output.at("atom_types").toTensor();
    std::cout << "atom types: (" << atom_types.size(0) << ", " << atom_types.size(1) << ")" << std::endl;
    float atom_masses[nat];
    for (int i = 0; i < nat; i++) {
        int atom_type = atom_types[i][0].item<int>();
	if (atom_type == 0) {
	    atom_masses[i] = 178.49;
	} else {
            atom_masses[i] = 15.999;
	}
    }
    float velocities[nat][3];
    memset(velocities, 0, nat*3*sizeof(float));
    float modified_acc[nat][3];
    for (int i = 0; i < nat; i++) {
        for (int j = 0; j < 3; j++) {
            modified_acc[i][j] = forces_tensor[i][j].item<float>() / atom_masses[i] - nvt_bath * velocities[i][j];
	    if (modified_acc[i][j] != modified_acc[i][j]) {
                modified_acc[i][j] = 0;
	    }
	}
    }
    float pos_fullstep[nat][3];
    for (int i = 0; i < nat; i++) {
        for (int j = 0; j < 3; j++) {
            pos_fullstep[i][j] = new_pos[i][j].item<float>() + dt * velocities[i][j] + 0.5 * dtdt * modified_acc[i][j];
	    if (pos_fullstep[i][j] != pos_fullstep[i][j]) {
                pos_fullstep[i][j] = new_pos[i][j].item<float>();
	    }
	}
    }
    float vel_halfstep[nat][3];
    for (int i = 0; i < nat; i++) {
        for (int j = 0; j < 3; j++) {
            vel_halfstep[i][j] = velocities[i][j] + 0.5 * dt * modified_acc[i][j];
	    if (vel_halfstep[i][j] != vel_halfstep[i][j]) {
                vel_halfstep[i][j] = 0;
	    }
	}
    }
    torch::Tensor pos_updated_after_physics = torch::from_blob(pos_fullstep, {nat, 3});
    float kB = 1.380649 * pow(10, -4) / 1.602176634;
    float e_kin_sum = 0;
    for (int i = 0; i < nat; i++) {
        float vel_sq_sum = 0;
        for (int j = 0; j < 3; j++) {
            vel_sq_sum += velocities[i][j] * velocities[i][j];
        }
        e_kin_sum += vel_sq_sum * atom_masses[i];
    }
    float e_kin_diff = 0.5 * (e_kin_sum - (3 * nat + 1) * kB * temp);
    float nvt_bath_halfstep = nvt_bath + 0.5 * dt * e_kin_diff / nvt_q;
    float e_kin_sum_halfstep = 0;
    for (int i = 0; i < nat; i++) {
        float vel_sq_sum_halfstep = 0;
        for (int j = 0; j < 3; j++) {
            vel_sq_sum_halfstep += vel_halfstep[i][j] * vel_halfstep[i][j];
        }
        e_kin_sum += vel_sq_sum_halfstep * atom_masses[i];
    }
    float e_kin_diff_halfstep = 0.5 * (e_kin_sum_halfstep - (3 * nat + 1) * kB * temp);
    nvt_bath = nvt_bath_halfstep + 0.5 * dt * e_kin_diff_halfstep / nvt_q;
    for (int i = 0; i < nat; i++) {
        for (int j = 0; j < 3; j++) {
            velocities[i][j] = vel_halfstep[i][j] + 0.5 * dt * (forces_tensor[i][j].item<float>() / atom_masses[i]);
        }
    }














    std::vector<torch::jit::IValue> inputs2;
    torch::Dict<std::string, torch::Tensor> dictionary2;

    for (DictEntry *entry = arrays; entry; entry = entry->next) {
        int rows = 1;
        int cols = 1;
        if (entry->nrows != 0) {
            rows = entry->nrows;
        }
        if (entry->ncols != 0) {
            cols = entry->ncols;
        }
        // std::cout << entry->key << std::endl;

        std::string str1 (entry->key);
        if (entry->data_t == data_s) {
            std::cout << (char**) (entry->data) << std::endl;
            int arr[cols];
            for (int j=0; j<cols; j++) {
                std::string type(((char**) (entry->data))[j]);
                if (type == "Hf") {
                    arr[j] = 0;
                } else if (type == "O") {
                    arr[j] = 1;
                }
            }
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            torch::Tensor t = torch::from_blob(arr, {cols, 1}, options=options).to(torch::kInt64);
            dictionary2.insert("atom_types", t);

            AtomGraph system(pos_updated_after_physics, cols, 5.0);
            torch::Tensor edges = system.edges();
            dictionary2.insert("edge_index", edges);
            std::cout << "SCOND ITME Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
            dictionary2.insert("pos", pos_updated_after_physics);
            // dictionary2.insert("forces", forces_tensor);
        }
    }

    torch::Tensor t3_2 = torch::randint(0, 1, {nat}).to(torch::kLong);
    dictionary2.insert("batch", t3_2);
    inputs2.push_back(dictionary2);

    // Execute the model and turn its output into a tensor.
    // module.eval();
    std::cout << "SECOND TEIM before forward" << std::endl;
    auto output2 = module(inputs2).toGenericDict();
    // auto output2 = module.forward(inputs2).toGenericDict();
    std::cout << "SECOND TIEM after forward" << std::endl;

    std::cout << "SECOND TIME keys\n--------------------------------------------------------" << std::endl;
    for (auto it = output2.begin(); it!=output2.end(); it++) {
        std::cout << it->key() << std::endl;
    }
    std::cout << "SECOND TIME keysdone\n--------------------------------------------------------" << std::endl;
    // size: (1857, 3)
    torch::Tensor forces_tensor2 = output2.at("forces").toTensor();
    std::cout << "SECOND TIME FORCES" << std::endl;
    for (int i = 0; i < nat; i++) {
        std::cout << i << ": " << forces_tensor2[i][0] << ", " << forces_tensor2[i][1] << ", " << forces_tensor2[i][2] << std::endl;
    }
    torch::Tensor total_energy_tensor2 = output2.at("total_energy").toTensor();
    torch::Tensor atomic_energy_tensor2 = output2.at("atomic_energy").toTensor();
    std::cout << "SECOND TIME energy dim: " << atomic_energy_tensor2.dim() << std::endl;
    std::cout << "SECOND TIME energy dim: " << atomic_energy_tensor2.size(0) << ", " << atomic_energy_tensor2.size(1) << std::endl;
    std::cout << "SECOND TIME ENERGIES" << std::endl;
    for (int i = 0; i < nat; i++) {
        std::cout << i << ": " << atomic_energy_tensor2[i][0] << std::endl;
    }
    // auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
    float atomic_energy_sum2 = atomic_energy_tensor2.sum().data_ptr<float>()[0];
    std::cout << "SECOND TIME Atomic energy sum: " << atomic_energy_sum2 << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
