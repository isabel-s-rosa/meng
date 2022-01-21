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
    AtomGraph(Node* nodes, int num_nodes, double r_cutoff)
    : m_nodes(nodes), m_num_nodes(num_nodes), m_r_cutoff(r_cutoff)
    {
        int num_edges = 0;
        for (int i = 0; i < m_num_nodes; i++) {
            V.push_back(num_edges);
            for (int j = 0; j < m_num_nodes; j++) {
                if ((dist_squared(m_nodes[i], m_nodes[j]) <= m_r_cutoff * m_r_cutoff) && i != j) {
                    E.push_back(j);
                    num_edges++;
                }
            }
        }
        V.push_back(num_edges);
    }

    double r_cutoff() const
    {
        return m_r_cutoff;
    }

    int num_nodes() const
    {
        return m_num_nodes;
    }

    Node* nodes() const
    {
        return m_nodes;
    }

    std::vector<int> vertices() const
    {
        return V;
    }

    std::vector<int> edges() const
    {
        return E;
    }

private:
    Node* m_nodes;
    int m_num_nodes;
    double m_r_cutoff;
    std::vector<int> V;
    std::vector<int> E;

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

    // std::cout << "ok" << std::endl;

    // std::cout << module.children().size() << std::endl;

    // std::cout << module.modules().size() << std::endl;

    // std::cout << module.dump_to_str(false, false, true) << std::endl;

    // for (auto thing : module.parameters()) {
    //     std::cout << thing << std::endl;
    //     std::cout << "here" << std::endl;
    // }

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
    // dictionary.insert("uh", torch::ones({1, 3, 224, 224}));
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(dictionary);

    for (DictEntry *entry = arrays; entry; entry = entry->next) {
        std::cout << entry->key << std::endl;
        int rows = 1;
        int cols = 1;
        if (entry->nrows != 0) {
            rows = entry->nrows;
        }
        if (entry->ncols != 0) {
            cols = entry->ncols;
        }
        if (entry->data_t == data_i) {
            std::cout << (int*) (entry->data) << std::endl;
            torch::Tensor t = torch::from_blob((int*) entry->data, {rows, cols});
            // for (int i=0; i<rows; i++) {
            //     for (int j=0; j<cols; j++) {
            //         std::cout << "i: " << i << " j: " << j << std::endl;
            //         std::cout << t[i][j] << std::endl;
            //     }
            // }
            dictionary.insert(entry->key, t);
        } else if (entry->data_t == data_f) {
            std::cout << (double*) (entry->data) << std::endl;
            torch::Tensor t = torch::from_blob((double*) entry->data, {rows, cols});
            // for (int i=0; i<rows; i++) {
            //     for (int j=0; j<cols; j++) {
            //         std::cout << "i: " << i << " j: " << j << std::endl;
            //         std::cout << t[i][j] << std::endl;
            //     }
            // }
            dictionary.insert(entry->key, t);
        } else if (entry->data_t == data_b) {
            std::cout << (int*) (entry->data) << std::endl;
            torch::Tensor t = torch::from_blob((int*) entry->data, {rows, cols});
            // for (int i=0; i<rows; i++) {
            //     for (int j=0; j<cols; j++) {
            //         std::cout << "i: " << i << " j: " << j << std::endl;
            //         std::cout << t[i][j] << std::endl;
            //     }
            // }
            dictionary.insert(entry->key, t);
        } else if (entry->data_t == data_s) {
            // std::cout << (char**) (entry->data) << std::endl;
            // int arr[rows * cols];
            // for (int j=0; j<cols; j++) {
            //     std::string type(((char**) (entry->data))[j]);
            //     std::cout << type << std::endl;
            //     if (type == "Hf") {
            //         arr[j] = 72;
            //     } else if (type == "O") {
            //         arr[j] = 8;
            //     }
            // }
            // for (int j=0; j<cols; j++) {
            //     std::cout << arr[j] << std::endl;
            // }
            int data[] = {72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 8, 8 };
            torch::Tensor t = torch::from_blob(data, {rows, cols}).to(torch::kInt64);
            for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                    std::cout << "i: " << i << " j: " << j << std::endl;
                    std::cout << t[i][j] << std::endl;
                }
            }
            dictionary.insert("atom_types", t);
        }
        // dictionary.insert(entry->key, &(entry->data));
    }
    // long data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0};
    // torch::Tensor t = torch::from_blob(data, {2, 32}).to(torch::kInt64);
    torch::Tensor t = torch::randint(0, 1, {2, 32}).to(torch::kLong);
    dictionary.insert("edge_index", t);
    inputs.push_back(dictionary);
    torch::Tensor t2 = dictionary.at("atom_types");
    std::cout << "ATOM TYPES:" << std::endl;
    for (int i=0; i<1; i++) {
        for (int j=0; j<32; j++) {
            std::cout << "i: " << i << " j: " << j << std::endl;
            std::cout << t[i][j] << std::endl;
        }
    }

    // 
    // // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    int num_atoms = 10;
    Node* positions = (Node*) malloc(num_atoms * sizeof(Node));
    for (int i = 0; i < num_atoms; i++) {
        Pos pos = {0,0,static_cast<double>(i)};
        Node node(pos);
        positions[i] = node;
        // std::cout << "here: " << positions[i].pos()[2] << std::endl;
    }

    double r_cutoff = 3.0;

    AtomGraph system(positions, num_atoms, r_cutoff);
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
