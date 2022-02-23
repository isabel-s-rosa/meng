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

    // int err_stat = extxyz_write_ll(stdout, nat, info, arrays);
    // if (! strcmp(argv[2], "T")) {
    //     std::cout << "written err_stat " << err_stat <<std::endl;
    // }
    // // Create a vector of inputs.
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
	std::string str3 ("forces");
        if (entry->data_t == data_f && (str1.compare(str2) == 0 || str1.compare(str3) == 0)) {
	    // std::cout << "HERE---------------------------------------------------" << std::endl;
	    // std::cout << entry->key << std::endl;
            // std::cout << (double*) (entry->data) << std::endl;
            torch::Tensor t = torch::from_blob((double*) entry->data, {rows, cols});
	    if (str1.compare(str2) == 0) {
                AtomGraph system(t, rows, 50.0);
		torch::Tensor edges = system.edges();
		dictionary.insert("edge_index", edges);
		std::cout << "Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
	    }
            // for (int i=0; i<rows; i++) {
            //     for (int j=0; j<cols; j++) {
            //         std::cout << "i: " << i << " j: " << j << std::endl;
            //         std::cout << t[i][j] << std::endl;
            //     }
            // }
            dictionary.insert(entry->key, t);
       } else if (entry->data_t == data_s) {
            std::cout << (char**) (entry->data) << std::endl;
            int arr[cols];
            for (int j=0; j<cols; j++) {
                std::string type(((char**) (entry->data))[j]);
                std::cout << type << std::endl;
                if (type == "Hf") {
                    arr[j] = 0;
                } else if (type == "O") {
                    arr[j] = 1;
                }
            }
            // for (int j=0; j<cols; j++) {
            //     std::cout << arr[j] << std::endl;
            // }
            // int data[] = {72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 8, 8 };
            // int data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
	    auto options = torch::TensorOptions().dtype(torch::kInt32);
            torch::Tensor t = torch::from_blob(arr, {cols, 1}, options=options).to(torch::kInt64);
	    std::cout << "printing atom types" << std::endl;
            for (int i=0; i<cols; i++) {
                for (int j=0; j<1; j++) {
                    std::cout << "i: " << i << " j: " << j << std::endl;
                    std::cout << t[i][j] << std::endl;
                }
            }
            dictionary.insert("atom_types", t);
        }

    //     if (entry->data_t == data_i) {
    //         std::cout << (int*) (entry->data) << std::endl;
    //         torch::Tensor t = torch::from_blob((int*) entry->data, {rows, cols});
    //         // for (int i=0; i<rows; i++) {
    //         //     for (int j=0; j<cols; j++) {
    //         //         std::cout << "i: " << i << " j: " << j << std::endl;
    //         //         std::cout << t[i][j] << std::endl;
    //         //     }
    //         // }
    //         dictionary.insert(entry->key, t);
    //     } else if (entry->data_t == data_f) {
    //         std::cout << (double*) (entry->data) << std::endl;
    //         torch::Tensor t = torch::from_blob((double*) entry->data, {rows, cols});
    //         // for (int i=0; i<rows; i++) {
    //         //     for (int j=0; j<cols; j++) {
    //         //         std::cout << "i: " << i << " j: " << j << std::endl;
    //         //         std::cout << t[i][j] << std::endl;
    //         //     }
    //         // }
    //         dictionary.insert(entry->key, t);
    //     } else if (entry->data_t == data_b) {
    //         std::cout << (int*) (entry->data) << std::endl;
    //         torch::Tensor t = torch::from_blob((int*) entry->data, {rows, cols});
    //         // for (int i=0; i<rows; i++) {
    //         //     for (int j=0; j<cols; j++) {
    //         //         std::cout << "i: " << i << " j: " << j << std::endl;
    //         //         std::cout << t[i][j] << std::endl;
    //         //     }
    //         // }
    //         dictionary.insert(entry->key, t);
    //     } else if (entry->data_t == data_s) {
    //         // std::cout << (char**) (entry->data) << std::endl;
    //         // int arr[rows * cols];
    //         // for (int j=0; j<cols; j++) {
    //         //     std::string type(((char**) (entry->data))[j]);
    //         //     std::cout << type << std::endl;
    //         //     if (type == "Hf") {
    //         //         arr[j] = 72;
    //         //     } else if (type == "O") {
    //         //         arr[j] = 8;
    //         //     }
    //         // }
    //         // for (int j=0; j<cols; j++) {
    //         //     std::cout << arr[j] << std::endl;
    //         // }
    //         int data[] = {72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 8, 8 };
    //         torch::Tensor t = torch::from_blob(data, {rows, cols}).to(torch::kInt64);
    //         for (int i=0; i<rows; i++) {
    //             for (int j=0; j<cols; j++) {
    //                 std::cout << "i: " << i << " j: " << j << std::endl;
    //                 std::cout << t[i][j] << std::endl;
    //             }
    //         }
    //         dictionary.insert("atom_types", t);
    //     }
    //     // dictionary.insert(entry->key, &(entry->data));
    }
    // long data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0};
    // torch::Tensor t = torch::from_blob(data, {2, 32}).to(torch::kInt64);
    // torch::Tensor t = torch::randint(0, 1, {2, 31902}).to(torch::kLong);
    // dictionary.insert("edge_index", t);

    // // num edges: 31902
    // // replacing with: 1857 ?
    // torch::Tensor t = torch::randint(0, 1, {2, 1857}).to(torch::kLong); //repl
    // dictionary.insert("edge_index", t);
    // torch::Tensor t2 = torch::randn({1857, 3});
    // dictionary.insert("pos", t2);
    // torch::Tensor t3 = torch::randint(0, 1, {1857}).to(torch::kLong);
    // dictionary.insert("batch", t3);
    // // torch::Tensor t4 = torch::randint(0, 1, {51}).to(torch::kLong);
    // // dictionary.insert("ptr", t4);
    // torch::Tensor t5 = torch::randn({50, 3, 3}).to(torch::kLong); //used to not be kLong
    // dictionary.insert("cell", t5);
    // torch::Tensor t6 = torch::randn({1857, 3});
    // dictionary.insert("forces", t6);
    // torch::Tensor t7 = torch::randint(0, 1, {1857, 3}).to(torch::kLong); //repl
    // dictionary.insert("edge_cell_shift", t7);
    // // torch::Tensor t8 = torch::randint(0, 1, {50, 3}).to(torch::kBool);
    // // dictionary.insert("pbc", t8);
    // // torch::Tensor t9 = torch::randn({50, 1});
    // // dictionary.insert("total_energy", t9);
    // // torch::Tensor t10 = torch::randn({300});
    // // dictionary.insert("stress", t10);
    // // torch::Tensor t11 = torch::randn({50});
    // // dictionary.insert("r_max", t11);
    // torch::Tensor t12 = torch::randint(0, 1, {1857, 1}).to(torch::kLong);
    // dictionary.insert("atom_types", t12);
    // // torch::Tensor t13 = torch::randn({1857, 2});
    // // dictionary.insert("node_attrs", t13);
    // // torch::Tensor t14 = torch::randn({1857, 16});
    // // dictionary.insert("node_features", t4);
    // // torch::Tensor t15 = torch::randn({1857, 3}); //repl
    // // dictionary.insert("edge_vectors", t15);
    // // torch::Tensor t16 = torch::randn({1857, 9}); //repl
    // // dictionary.insert("edge_attrs", t16);
    // // torch::Tensor t17 = torch::randn({1857}); //repl
    // // dictionary.insert("edge_lengths", t17);
    // // torch::Tensor t18 = torch::randn({1857, 20}); //repl
    // // dictionary.insert("edge_embedding", t18);
    // torch::Tensor t19 = torch::randn({1857, 1});
    // dictionary.insert("atomic_energy", t19);
    // inputs.push_back(dictionary);

    // num edges: 31902
    // replacing with: 1857 ?
    // replacing with: 32 ?
    // torch::Tensor t = torch::randint(0, 1, {2, 50}).to(torch::kLong); //repl
    // dictionary.insert("edge_index", t);
    // torch::Tensor t2 = torch::randn({32, 3});
    // dictionary.insert("pos", t2);
    torch::Tensor t3 = torch::randint(0, 1, {nat}).to(torch::kLong);
    dictionary.insert("batch", t3);
    // torch::Tensor t5 = torch::randn({50, 3, 3}).to(torch::kLong); //used to not be kLong
    // dictionary.insert("cell", t5);
    // torch::Tensor t6 = torch::randn({32, 3});
    // dictionary.insert("forces", t6);
    // torch::Tensor t7 = torch::randint(0, 1, {32, 3}).to(torch::kLong); //repl
    // dictionary.insert("edge_cell_shift", t7);
    // torch::Tensor t12 = torch::randint(0, 1, {32, 1}).to(torch::kLong);
    // dictionary.insert("atom_types", t12);
    // torch::Tensor t19 = torch::randn({32, 1});
    // dictionary.insert("atomic_energy", t19);
    inputs.push_back(dictionary);
    // torch::Tensor t2 = dictionary.at("atom_types");
    // std::cout << "ATOM TYPES:" << std::endl;
    // for (int i=0; i<32; i++) {
    //     for (int j=0; j<1; j++) {
    //         std::cout << "i: " << i << " j: " << j << std::endl;
    //         std::cout << t2[i][j] << std::endl;
    //     }
    // }

    // 
    // // Execute the model and turn its output into a tensor.
    // std::cout << "before eval" << std::endl;
    module.eval();
    std::cout << "before forward" << std::endl;
    auto output = module.forward(inputs).toGenericDict();
    std::cout << "after forward" << std::endl;

    std::cout << "keys\n--------------------------------------------------------" << std::endl;
    for (auto it = output.begin(); it!=output.end(); it++) {
        std::cout << it->key() << std::endl;
    }
    std::cout << "keysdone\n--------------------------------------------------------" << std::endl;
    // size: (1857, 3)
    torch::Tensor forces_tensor = output.at("forces").toTensor();
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

    // int num_atoms = 10;
    // Node* positions = (Node*) malloc(num_atoms * sizeof(Node));
    // for (int i = 0; i < num_atoms; i++) {
    //     Pos pos = {0,0,static_cast<double>(i)};
    //     Node node(pos);
    //     positions[i] = node;
    //     // std::cout << "here: " << positions[i].pos()[2] << std::endl;
    // }

    // double r_cutoff = 3.0;

    // AtomGraph system(positions, num_atoms, r_cutoff);
    // std::vector<int> V = system.vertices();
    // std::vector<int> E = system.edges();
    // for (int i = 0; i < V.size() - 1; i++) {
    //     std::cout << "V: " << i << std::endl;
    //     std::cout << "Index into E: " << V[i] << std::endl;
    //     for (int j = V[i]; j < V[i+1]; j++) {
    //         std::cout << "E: " << E[j] << std::endl;
    //     }
    //     std::cout << "--------------------------" << std::endl;
    // }
}

////////////////////////////////////////////////////////////////////////////////
