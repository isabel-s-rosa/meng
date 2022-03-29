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

typedef struct { int pos_x0, left_slope, pos_x1, right_slope; } Zoid;

// x y z right and left may not matter, i think we want to cut in by 3r from
// most extreme atoms in each dimension each time instead of using a boundary
// in 3D space to define cuts bc something might cross that boundary
void  trapezoid(torch::jit::script::Module module,
                int t0,
                int t1,
                int nat1,
                /*torch::Tensor atoms1,*/
		float* atoms1,
                int* atom_types1,
                /*torch::Tensor &atoms_ret,*/
		float* atoms_ret,
                int* atom_types_ret,
                int* timestep_ret,
                /*float x_left,
                float x_right,
                float y_left,
                float y_right,
                float z_left,
                float z_right,*/
                bool x_left_cut,
                bool x_right_cut,
                bool y_left_cut,
                bool y_right_cut,
                bool z_left_cut,
                bool z_right_cut)
{
    std::cout << "t0: " << t0 << std::endl;
    std::cout << "t1: " << t1 << std::endl;
    /*std::cout << "x left: " << x_left << std::endl;
    std::cout << "x right: " << x_right << std::endl;
    std::cout << "y left: " << y_left << std::endl;
    std::cout << "y right: " << y_right << std::endl;
    std::cout << "z left: " << z_left << std::endl;
    std::cout << "z right: " << z_right << std::endl;*/
    int delta_t = t1 - t0;

    if (delta_t == 1) {
        std::cout << "base case!" << std::endl;
        // run nequip on 
        std::vector<torch::jit::IValue> inputs2;
        torch::Dict<std::string, torch::Tensor> dictionary2;
        // for (int i = 0; i < nat1; i++) {
	//     std::cout << "atoms1: " << atoms1[i][0] << ", " << atoms1[i][1] << ", " << atoms1[i][2] << std::endl;
        // }

        torch::Tensor pos_base = torch::from_blob(atoms1, {nat1, 3});
        // torch::Tensor pos_base = torch::randint(0, 1, {nat1, 3}).to(torch::kFloat);
        // std::cout << "poses at start with t0: " << t0 << ", t1: " << t1 << std::endl;
        // for (int i = 0; i < nat1; i++) {
        //     std::cout << i << ": " << atoms1[i][0].item<float>() << ", " << atoms1[i][1].item<float>() << ", " << atoms1[i][2].item<float>() << std::endl;
        // }
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor t = torch::from_blob(atom_types1, {nat1, 1}, options=options).to(torch::kInt64);
        dictionary2.insert("atom_types", t);

	// std::cout << "Pos dim 0: " << pos_base.size(0) << ", dim 1: " << pos_base.size(1) << std::endl;
	std::cout << "nat: " << nat1 << std::endl;
        AtomGraph system(pos_base, nat1, 3.0);
        torch::Tensor edges = system.edges();
        dictionary2.insert("edge_index", edges);
	std::cout << "Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
        dictionary2.insert("pos", pos_base);

        torch::Tensor t3 = torch::randint(0, 1, {nat1}).to(torch::kLong);
        dictionary2.insert("batch", t3);
        inputs2.push_back(dictionary2);

	std::cout << "calling forward" << std::endl;
        auto output2 = module(inputs2).toGenericDict();
	std::cout << "after forward" << std::endl;

        torch::Tensor new_pos = output2.at("pos").toTensor().detach();
        torch::Tensor forces_tensor = output2.at("forces").toTensor().detach();

        // physics stuff
        float temp = 300.0;
        float nvt_q = 334.0;
        float dt = 0.5 * pow(10, -15); //femtoseconds?
        float dtdt = pow(dt, 2);
        float nvt_bath = 0.0;
        torch::Tensor atom_types = output2.at("atom_types").toTensor().detach();
        // std::cout << "atom types: (" << atom_types.size(0) << ", " << atom_types.size(1) << ")" << std::endl;
        float atom_masses[nat1];
        for (int i = 0; i < nat1; i++) {
            int atom_type = atom_types[i][0].item<int>();
            if (atom_type == 0) {
                atom_masses[i] = 178.49;
            } else {
                atom_masses[i] = 15.999;
            }
        }
        float velocities[nat1][3];
        memset(velocities, 0, nat1*3*sizeof(float));
        float modified_acc[nat1][3];
        for (int i = 0; i < nat1; i++) {
            for (int j = 0; j < 3; j++) {
                modified_acc[i][j] = forces_tensor[i][j].item<float>() / atom_masses[i] - nvt_bath * velocities[i][j];
                if (modified_acc[i][j] != modified_acc[i][j]) {
                    modified_acc[i][j] = 0;
                }
            }
        }
	float x_left, x_right, y_left, y_right, z_left, z_right;
        float pos_fullstep[nat1][3];
        for (int i = 0; i < nat1; i++) {
            for (int j = 0; j < 3; j++) {
                pos_fullstep[i][j] = new_pos[i][j].item<float>() + dt * velocities[i][j] + 0.5 * dtdt * modified_acc[i][j];
                if (pos_fullstep[i][j] != pos_fullstep[i][j]) {
                    pos_fullstep[i][j] = new_pos[i][j].item<float>();
                }
            }
	    if (i == 0) {
	        x_left = pos_fullstep[i][0];
	        y_left = pos_fullstep[i][1];
	        z_left = pos_fullstep[i][2];
	        x_right = pos_fullstep[i][0];
	        y_right = pos_fullstep[i][1];
	        z_right = pos_fullstep[i][2];
	    } else {
	        x_left = std::min(x_left, pos_fullstep[i][0]);
	        y_left = std::min(y_left, pos_fullstep[i][1]);
	        z_left = std::min(z_left, pos_fullstep[i][2]);
	        x_right = std::max(x_right, pos_fullstep[i][0]);
	        y_right = std::max(y_right, pos_fullstep[i][1]);
	        z_right = std::max(z_right, pos_fullstep[i][2]);
	    }
        }
        float vel_halfstep[nat1][3];
        for (int i = 0; i < nat1; i++) {
            for (int j = 0; j < 3; j++) {
                vel_halfstep[i][j] = velocities[i][j] + 0.5 * dt * modified_acc[i][j];
                if (vel_halfstep[i][j] != vel_halfstep[i][j]) {
                    vel_halfstep[i][j] = 0;
                }
            }
        }
        torch::Tensor pos_updated_after_physics = torch::from_blob(pos_fullstep, {nat1, 3});
        float kB = 1.380649 * pow(10, -4) / 1.602176634;
        float e_kin_sum = 0;
        for (int i = 0; i < nat1; i++) {
            float vel_sq_sum = 0;
            for (int j = 0; j < 3; j++) {
                vel_sq_sum += velocities[i][j] * velocities[i][j];
            }
            e_kin_sum += vel_sq_sum * atom_masses[i];
        }
        float e_kin_diff = 0.5 * (e_kin_sum - (3 * nat1 + 1) * kB * temp);
        float nvt_bath_halfstep = nvt_bath + 0.5 * dt * e_kin_diff / nvt_q;
        float e_kin_sum_halfstep = 0;
        for (int i = 0; i < nat1; i++) {
            float vel_sq_sum_halfstep = 0;
            for (int j = 0; j < 3; j++) {
                vel_sq_sum_halfstep += vel_halfstep[i][j] * vel_halfstep[i][j];
            }
            e_kin_sum += vel_sq_sum_halfstep * atom_masses[i];
        }
        float e_kin_diff_halfstep = 0.5 * (e_kin_sum_halfstep - (3 * nat1 + 1) * kB * temp);
        nvt_bath = nvt_bath_halfstep + 0.5 * dt * e_kin_diff_halfstep / nvt_q;
        for (int i = 0; i < nat1; i++) {
            for (int j = 0; j < 3; j++) {
                velocities[i][j] = vel_halfstep[i][j] + 0.5 * dt * (forces_tensor[i][j].item<float>() / atom_masses[i]);
            }
        }

	float new_x_left = x_left;
	float new_x_right = x_right;
	float new_y_left = y_left;
	float new_y_right = y_right;
	float new_z_left = z_left;
	float new_z_right = z_right;
	if (x_left_cut) {
            new_x_left = x_left + 3 * 3.0;
	}
	if (x_right_cut) {
            new_x_right = x_right - 3 * 3.0;
	}
	if (y_left_cut) {
            new_y_left = y_left + 3 * 3.0;
	}
	if (y_right_cut) {
            new_y_right = y_right - 3 * 3.0;
	}
	if (z_left_cut) {
            new_z_left = z_left + 3 * 3.0;
	}
	if (z_right_cut) {
            new_z_right = z_right - 3 * 3.0;
	}
	// 3d coords + time step after which it gets cut + atom type
	// float* to_return = new float[nat1 * 3];
	for (int i = 0; i < nat1; i++) {
            atoms_ret[i * 3 + 0] = pos_fullstep[i][0];
            atoms_ret[i * 3 + 1] = pos_fullstep[i][1];
            atoms_ret[i * 3 + 2] = pos_fullstep[i][2];
	    if (((pos_fullstep[i][0] >= new_x_left &&
                  pos_fullstep[i][0] <= new_x_right) ||
		 pos_fullstep[i][0] != pos_fullstep[i][0]) &&
	        ((pos_fullstep[i][1] >= new_y_left &&
	          pos_fullstep[i][1] <= new_y_right) ||
		 pos_fullstep[i][1] != pos_fullstep[i][1]) &&
	        ((pos_fullstep[i][2] >= new_z_left &&
	          pos_fullstep[i][2] <= new_z_right) ||
		 pos_fullstep[i][2] != pos_fullstep[i][2])) {

                // would not have gotten cut next step
		// not on the edge
		timestep_ret[i] = t1;
		// std::cout << "not on edge, t1: " << t1 << std::endl;
                // to_return[i][3] = static_cast<float>(t1);
		//     std::cout << "returning: " << to_return[i][3] << std::endl;

	    } else {
                // woudl have gotten cut next step
		// on edge
		// std::cout << "adding something to edge, only possible w space cut" << std::endl;
		// to_return[i][3] = static_cast<float>(t0);
		timestep_ret[i] = t0;
	    }
	    atom_types_ret[i] = atom_types1[i];
	    // to_return[i][4] = static_cast<float>(atom_types1[i]);
	}
        // torch::Tensor return_tensor = torch::from_blob(to_return, {nat1, 5});
	// atoms_ret = torch::from_blob(to_return, {nat1, 3});
        // std::cout << "poses at end with t0: " << t0 << ", t1: " << t1 << std::endl;
        // for (int i = 0; i < nat1; i++) {
        //     std::cout << i << ": " << atoms_ret[i][0].item<float>() << ", " << atoms_ret[i][1].item<float>() << ", " << atoms_ret[i][2].item<float>() << std::endl;
        // }
	std::cout << "returning from base case" << std::endl;
	// return return_tensor;
    } else {
        // abt 50 edges each ??
        // nat >> delta_t * 3 * r??
	// nat > 50 * delta_t + buffer (we expect to lose abt 50 atoms per step ?)
	// which axis (xyz) to use ?
	// calc xyz max min, calc xyz average
	// if enough atoms have xyz < xyz ave - delta_t * 3 * r and xyz > xyz ave + delta_t * 3 * r, do xyz cut
	if (t0 == 0 && t1 == 5 && nat1 == 768) {
            std::cout << "checking if space cut possible" << std::endl;
	    std::vector<float> xs;
	    for (int i = 0; i < nat1; i++) {
                if (atoms1[i * 3 + 0] != atoms1[i * 3 + 0]) {
                    // nan
		} else {
                    xs.push_back(atoms1[i * 3 + 0]);
		}
	    }
            nth_element(xs.begin(), xs.begin() + nat1 / 2, xs.end());
	    float med_x = (float) xs[nat1 / 2];
	    std::cout << "x med: " << med_x << std::endl;
	    int nat_below_x_ave_top = 0;
	    int nat_above_x_ave_top = 0;
	    int nat_below_x = 0;
	    int nat_above_x = 0;
	    float* atoms_below_x = new float[nat1 * 3];
	    float* atoms_above_x = new float[nat1 * 3];
	    int* atom_types_below_x = new int[nat1];
	    int* atom_types_above_x = new int[nat1];
	    for (int i = 0; i < nat1; i++) {
                if (atoms1[i * 3 + 0] <= med_x) {
	    	    atoms_below_x[nat_below_x * 3 + 0] = atoms1[i * 3 + 0];
	    	    atoms_below_x[nat_below_x * 3 + 1] = atoms1[i * 3 + 1];
	    	    atoms_below_x[nat_below_x * 3 + 2] = atoms1[i * 3 + 2];
	    	    atom_types_below_x[nat_below_x] = atom_types1[i];
                    nat_below_x++;
		    if (atoms1[i * 3 + 0] < med_x - delta_t * 3 * 3.0) {
                        nat_below_x_ave_top++;
		    }
	        } else {
	    	    atoms_above_x[nat_above_x * 3 + 0] = atoms1[i * 3 + 0];
	    	    atoms_above_x[nat_above_x * 3 + 1] = atoms1[i * 3 + 1];
	    	    atoms_above_x[nat_above_x * 3 + 2] = atoms1[i * 3 + 2];
	    	    atom_types_above_x[nat_above_x] = atom_types1[i];
                    nat_above_x++;
                    if (atoms1[i * 3 + 0] > med_x + delta_t * 3 * 3.0) {
                        nat_above_x_ave_top++;
		    }
	        }
	    }
	    std::cout << "nat below x ave top: " << nat_below_x_ave_top << std::endl;
	    std::cout << "nat above x ave top: " << nat_above_x_ave_top << std::endl;
	    if (nat_below_x_ave_top >= 4 /*arbitrary, should change*/ && nat_above_x_ave_top >= 4) {
                // wide enough in x dim
	        std::cout << "recursing left trap, nat below: " << nat_below_x << std::endl;
	        float* atoms_left_final = new float[nat_below_x * 3];
	        int* atom_types_left_final = new int[nat_below_x];
	        int* atom_times_left_final = new int[nat_below_x];
	        trapezoid(module,
	                  t0,
	                  t1,
	                  nat_below_x,
	                  atoms_below_x,
	                  atom_types_below_x,
	                  atoms_left_final,
	                  atom_types_left_final,
	                  atom_times_left_final,
	                  /*x_left,
	                  x_right,
	                  y_left,
	                  y_right,
	                  z_left,
	                  z_right,*/
	                  x_left_cut,
	                  true,
	                  y_left_cut,
	                  y_right_cut,
	                  z_left_cut,
	                  z_right_cut);
	        std::cout << "done left trap" << std::endl;
                // std::cout << "poses of left trap, t0: " << t0 << ", t1: " << t1 << std::endl;
                // for (int i = 0; i < nat_below_x; i++) {
                //     std::cout << i << ": " << atoms_left_final[i * 3 + 0] << ", " << atoms_left_final[i * 3 + 1] << ", " << atoms_left_final[i * 3 + 2] << std::endl;
	        // }
	        std::cout << "recursing right trap, nat above: " << nat_above_x << std::endl;
	        float* atoms_right_final = new float[nat_above_x * 3];
	        int* atom_types_right_final = new int[nat_above_x];
	        int* atom_times_right_final = new int[nat_above_x];
	        trapezoid(module,
	                  t0,
	                  t1,
	                  nat_above_x,
	                  atoms_above_x,
	                  atom_types_above_x,
	                  atoms_right_final,
	                  atom_types_right_final,
	                  atom_times_right_final,
	                  /*x_left,
	                  x_right,
	                  y_left,
	                  y_right,
	                  z_left,
	                  z_right,*/
	                  true,
	                  x_right_cut,
	                  y_left_cut,
	                  y_right_cut,
	                  z_left_cut,
	                  z_right_cut);
	        std::cout << "done right trap" << std::endl;
                // std::cout << "poses of left trap, t0: " << t0 << ", t1: " << t1 << std::endl;
                // for (int i = 0; i < nat_below_x; i++) {
                //     std::cout << i << ": " << atoms_left_final[i * 3 + 0] << ", " << atoms_left_final[i * 3 + 1] << ", " << atoms_left_final[i * 3 + 2] << std::endl;
	        // }
	        return;
	    }
	}

	// other dimension cuts
	// float y_ave = 0;
	// for (int i = 0; i < nat1; i++) {
	//     y_ave += atoms1[i * 3 + 1];
	// }
	// y_ave /= nat1;
	// int nat_below_y_ave_top, nat_above_y_ave_top = 0;
	// int nat_below_y, nat_above_y = 0;
	// float* atoms_below_y = new float[nat1 * 3];
	// float* atoms_above_y = new float[nat1 * 3];
	// for (int i = 0; i < nat1; i++) {
        //     if (atoms1[i * 3 + 1] < y_ave - delta_t * 3 * 5.0) {
	// 	atoms_below_y[nat_below_y * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_below_y[nat_below_y * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_below_y[nat_below_y * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_below_y++;
        //         nat_below_y_ave_top++;
	//     } else if (atoms1[i * 3 + 1] <= y_ave) {
	// 	atoms_below_y[nat_below_y * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_below_y[nat_below_y * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_below_y[nat_below_y * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_below_y++;
	//     } else if (atoms1[i * 3 + 1] > y_ave + delta_t * 3 * 5.0) {
	// 	atoms_above_y[nat_above_y * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_above_y[nat_above_y * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_above_y[nat_above_y * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_above_y++;
        //         nat_above_y_ave_top++;
	//     } else {
	// 	atoms_above_y[nat_above_y * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_above_y[nat_above_y * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_above_y[nat_above_y * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_above_y++;
	//     }
	// }
	// if (nat_below_y_ave_top >= 5 /*arbitrary, should change*/ && nat_above_y_ave_top >= 5) {
        //     // wide enough in y dim
	//     return;
	// }

	// float z_ave = 0;
	// for (int i = 0; i < nat1; i++) {
	//     z_ave += atoms1[i * 3 + 2];
	// }
	// z_ave /= nat1;
	// int nat_below_z_ave_top, nat_above_z_ave_top = 0;
	// int nat_below_z, nat_above_z = 0;
	// float* atoms_below_z = new float[nat1 * 3];
	// float* atoms_above_z = new float[nat1 * 3];
	// for (int i = 0; i < nat1; i++) {
        //     if (atoms1[i * 3 + 2] < z_ave - delta_t * 3 * 5.0) {
	// 	atoms_below_z[nat_below_z * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_below_z[nat_below_z * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_below_z[nat_below_z * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_below_z++;
        //         nat_below_z_ave_top++;
	//     } else if (atoms1[i * 3 + 2] <= z_ave) {
	// 	atoms_below_z[nat_below_z * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_below_z[nat_below_z * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_below_z[nat_below_z * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_below_z++;
	//     } else if (atoms1[i * 3 + 2] > z_ave + delta_t * 3 * 5.0) {
	// 	atoms_above_z[nat_above_z * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_above_z[nat_above_z * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_above_z[nat_above_z * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_above_z++;
        //         nat_above_z_ave_top++;
	//     } else {
	// 	atoms_above_z[nat_above_z * 3 + 0] = atoms1[i * 3 + 0];
	// 	atoms_above_z[nat_above_z * 3 + 1] = atoms1[i * 3 + 1];
	// 	atoms_above_z[nat_above_z * 3 + 2] = atoms1[i * 3 + 2];
        //         nat_above_z++;
	//     }
	// }
	// if (nat_below_z_ave_top >= 5 /*arbitrary, should change*/ && nat_above_z_ave_top >= 5) {
        //     // wide enough in z dim
	//     return;
	// }

	// TIME CUT
        int halftime = delta_t / 2;
	// bottom zoid:
	// float** atoms_copy = new float*[nat1];
	// int* atom_types_copy = new int[nat1];
	// for (int i = 0; i < nat1; i++) {
        //     atom_types_copy[i] = atom_types1[i];
	//     atoms_copy[i] = new float[3];
	//     atoms_copy[0] = atoms1[0];
	//     atoms_copy[1] = atoms1[1];
	//     atoms_copy[2] = atoms1[2];
	// }
	std::cout << "recursing bottom trap" << std::endl;
	// torch::Tensor atoms_bottom;
	float* atoms_bottom = new float[nat1 * 3];
	int* atom_types_bottom = new int[nat1];
	int* atom_times_bottom = new int[nat1];
	trapezoid(module,
	          t0,
	          t0 + halftime,
	          nat1,
	          atoms1,
	          atom_types1,
	          atoms_bottom,
	          atom_types_bottom,
	          atom_times_bottom,
	          /*x_left,
	          x_right,
	          y_left,
	          y_right,
	          z_left,
	          z_right,*/
	          x_left_cut,
	          x_right_cut,
	          y_left_cut,
	          y_right_cut,
	          z_left_cut,
	          z_right_cut);
	std::cout << "done bottom trap" << std::endl;
        // std::cout << "poses of bottom trap, t0: " << t0 << ", t1: " << t1 << std::endl;
        // for (int i = 0; i < nat1; i++) {
        //     std::cout << i << ": " << atoms_bottom[i * 3 + 0] << ", " << atoms_bottom[i * 3 + 1] << ", " << atoms_bottom[i * 3 + 2] << std::endl;
	// }
	// top zoid:
	int nat_left = 0;
	float* atoms_left = new float[nat1 * 3];
	int* atom_types_left = new int[nat1];
	// float* merged_atom_info = new float[nat1 * 3];
	int j = 0;
        // std::cout << "poses before upper, t0: " << t0 << ", t1: " << t1 << std::endl;
	for (int k = 0; k < nat1; k++) {
	    if ( atom_times_bottom[k] == t0 + halftime ) {
                // top row of atoms, will not get cut on next time step
	        // std::cout << k << ": " << atoms_bottom[k * 3 + 0] << ", " << atoms_bottom[k * 3 + 1] << ", " << atoms_bottom[k * 3 + 2] << std::endl;
	        atoms_left[nat_left * 3 + 0] = atoms_bottom[k * 3 + 0];
	        atoms_left[nat_left * 3 + 1] = atoms_bottom[k * 3 + 1];
	        atoms_left[nat_left * 3 + 2] = atoms_bottom[k * 3 + 2];
	        // std::cout << k << ": " << atoms_left[nat_left * 3 + 0] << ", " << atoms_left[nat_left * 3 + 1] << ", " << atoms_left[nat_left * 3 + 2] << std::endl;
	        atom_types_left[nat_left] = atom_types_bottom[k];
	        nat_left++;
	    } else {
                // edges, already got cut or will get cut next
	        // check this, maybe ones that will get cut on next time step should be in atoms_left?
                atoms_ret[j * 3 + 0] = atoms_bottom[k * 3 + 0];
                atoms_ret[j * 3 + 1] = atoms_bottom[k * 3 + 1];
                atoms_ret[j * 3 + 2] = atoms_bottom[k * 3 + 2];
	        atom_types_ret[j] = atom_types_bottom[k];
	        timestep_ret[j] = atom_times_bottom[k];
	        j++;
	    }
	}
	std::cout << "atoms left at base of upper trap: " << nat_left << std::endl;
	// for (int i = 0; i < nat1; i++) {
	//     if (atoms1[i][0] >= new_x_left &&
        //         atoms1[i][0] <= new_x_right &&
	//         atoms1[i][1] >= new_y_left &&
	//         atoms1[i][1] <= new_y_right &&
	//         atoms1[i][2] >= new_z_left &&
	//         atoms1[i][2] <= new_z_right) {

        //         atoms_left[nat_left] = new float[3];
	//         atoms_left[nat_left][0] = atoms1[i][0];
	//         atoms_left[nat_left][1] = atoms1[i][1];
	//         atoms_left[nat_left][2] = atoms1[i][2];
	//         atom_types_left[nat_left] = atom_types1[i];
	//         nat_left++;

	//     }
	// }
	std::cout << "recursing top trap" << std::endl;
        // torch::Tensor pos_base2 = torch::from_blob(atoms_left, {nat_left, 3});
	// torch::Tensor atoms_top;
	float* atoms_top = new float[nat_left * 3];
	int* atom_types_top = new int[nat_left];
	int* atom_times_top = new int[nat_left];
	/*float new_x_left = x_left;
	float new_x_right = x_right;
	float new_y_left = y_left;
	float new_y_right = y_right;
	float new_z_left = z_left;
	float new_z_right = z_right;
	if (x_left_cut) {
            new_x_left = x_left + 3 * 5.0 * halftime;
	}
	if (x_right_cut) {
            new_x_right = x_right - 3 * 5.0 * halftime;
	}
	if (y_left_cut) {
            new_y_left = y_left + 3 * 5.0 * halftime;
	}
	if (y_right_cut) {
            new_y_right = y_right - 3 * 5.0 * halftime;
	}
	if (z_left_cut) {
            new_z_left = z_left + 3 * 5.0 * halftime;
	}
	if (z_right_cut) {
            new_z_right = z_right - 3 * 5.0 * halftime;
	}*/
	trapezoid(module,
	          t0 + halftime,
	          t1,
	          nat_left,
	          atoms_left,
	          atom_types_left,
	          atoms_top,
	          atom_types_top,
	          atom_times_top,
	          /*new_x_left,
	          new_x_right,
	          new_y_left,
	          new_y_right,
	          new_z_left,
	          new_z_right,*/
	          x_left_cut,
	          x_right_cut,
	          y_left_cut,
	          y_right_cut,
	          z_left_cut,
	          z_right_cut);
	std::cout << "done top trap" << std::endl;
	for (int i = 0; i < nat_left;  i++) {
            atoms_ret[j * 3 + 0] = atoms_top[i * 3 + 0];
            atoms_ret[j * 3 + 1] = atoms_top[i * 3 + 1];
            atoms_ret[j * 3 + 2] = atoms_top[i * 3 + 2];
	    atom_types_ret[j] = atom_types_top[i];
	    timestep_ret[j] = atom_times_top[i];
	    j++;
	}
	std::cout << "nat total: " << nat1 << std::endl;
	std::cout << "j: " << j << std::endl;
	std::cout << "returning non-base" << std::endl;
        // std::cout << "poses of returned trap b4 tensor, t0: " << t0 << ", t1: " << t1 << std::endl;
        // for (int i = 0; i < nat1; i++) {
        //     std::cout << i << ": " << merged_atom_info[i][0] << ", " << merged_atom_info[i][1] << ", " << merged_atom_info[i][2] << std::endl;
	// }
            // atoms_ret = torch::from_blob(merged_atom_info, {nat1, 3});
        // std::cout << "poses of returned trap, t0: " << t0 << ", t1: " << t1 << std::endl;
        // for (int i = 0; i < nat1; i++) {
        //     std::cout << i << ": " << atoms_ret[i][0].item<float>() << ", " << atoms_ret[i][1].item<float>() << ", " << atoms_ret[i][2].item<float>() << std::endl;
	// }
	// return to_return;
    }
}

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

    int num_steps = atoi(argv[3]);
    std::cout << "num steps: " << num_steps << std::endl;
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

    torch::Tensor pos_param;
    int* arr = new int[nat];
    for (DictEntry *entry = arrays; entry; entry = entry->next) {
        int rows = 1;
        int cols = 1;
        if (entry->nrows != 0) {
            rows = entry->nrows;
        }
        if (entry->ncols != 0) {
            cols = entry->ncols;
        }
	std::string str1 (entry->key);
	std::string str2 ("pos");
        if (entry->data_t == data_f && str1.compare(str2) == 0) {
            pos_param = torch::from_blob((float*) entry->data, {rows, cols}).to(torch::kFloat);
	    if (str1.compare(str2) == 0) {
                AtomGraph system(pos_param, nat, 3.0);
		torch::Tensor edges = system.edges();
		dictionary.insert("edge_index", edges);
		std::cout << "Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
	    }
            dictionary.insert(entry->key, pos_param);
       } else if (entry->data_t == data_s) {
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

    // // Execute the model and turn its output into a tensor.
    // // module.eval();
    // std::cout << "before forward" << std::endl;
    // auto output = module(inputs).toGenericDict();
    // std::cout << "after forward" << std::endl;

    // std::cout << "keys\n--------------------------------------------------------" << std::endl;
    // for (auto it = output.begin(); it!=output.end(); it++) {
    //     std::cout << it->key() << std::endl;
    // }
    // std::cout << "keysdone\n--------------------------------------------------------" << std::endl;
    // // size: (1857, 3)
    // torch::Tensor forces_tensor = output.at("forces").toTensor().detach();
    // // std::cout << "FORCES" << std::endl;
    // // for (int i = 0; i < nat; i++) {
    // //     std::cout << i << ": " << forces_tensor[i][0] << ", " << forces_tensor[i][1] << ", " << forces_tensor[i][2] << std::endl;
    // // }
    // torch::Tensor total_energy_tensor = output.at("total_energy").toTensor();
    // torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();
    // // std::cout << "energy dim: " << atomic_energy_tensor.dim() << std::endl;
    // // std::cout << "energy dim: " << atomic_energy_tensor.size(0) << ", " << atomic_energy_tensor.size(1) << std::endl;
    // // std::cout << "ENERGIES" << std::endl;
    // // for (int i = 0; i < nat; i++) {
    // //     std::cout << i << ": " << atomic_energy_tensor[i][0] << std::endl;
    // // }
    // // auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
    // float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];
    // std::cout << "Atomic energy sum: " << atomic_energy_sum << std::endl;

    float* atoms1 = new float[nat * 3];
    /*float x_left = pos_param[0][0].item<float>();
    float x_right = x_left;
    float y_left = pos_param[0][1].item<float>();
    float y_right = y_left;
    float z_left = pos_param[0][2].item<float>();
    float z_right = z_left;*/
    for (int i = 0; i < nat; i++) {
        atoms1[i * 3 + 0] = pos_param[i][0].item<float>();
        atoms1[i * 3 + 1] = pos_param[i][1].item<float>();
        atoms1[i * 3 + 2] = pos_param[i][2].item<float>();
        // std::cout << "pos param: " << pos_param[i][0].item<float>() << ", " << pos_param[i][1].item<float>() << ", " << pos_param[i][2].item<float>() << std::endl;
        // std::cout << "atoms1: " << atoms1[i][0] << ", " << atoms1[i][1] << ", " << atoms1[i][2] << std::endl;
        /*x_left = std::min(atoms1[i * 3 + 0], x_left);
        y_left = std::min(atoms1[i * 3 + 1], y_left);
        y_left = std::min(atoms1[i * 3 + 2], z_left);
        x_right = std::max(atoms1[i * 3 + 0], x_right);
        y_right = std::max(atoms1[i * 3 + 1], y_right);
        y_right = std::max(atoms1[i * 3 + 2], z_right);*/
    }
    std::cout << "starting zoid lol" << std::endl;
    std::cout << "num steps again: " << num_steps << std::endl;
    // module.eval();
    float* atoms_answer = new float[nat];
    int* atoms_types = new int[nat];
    int* atoms_times = new int[nat];
    trapezoid(module,
              0,
              num_steps,
              nat,
              atoms1,
              arr,
              atoms_answer,
              atoms_types,
              atoms_times,
              /*x_left,
              x_right,
              y_left,
              y_right,
              z_left,
              z_right,*/
              false,
              false,
              false,
              false,
              false,
              false);
    std::cout << "ZOID DONE ------------------------------------------------------------------" << std::endl;

    // for (int step = 0; step < num_steps; step++) {
    //     torch::Tensor new_pos = output.at("pos").toTensor().detach();

    //     // physics stuff
    //     float temp = 300.0;
    //     float nvt_q = 334.0;
    //     float dt = 0.5 * pow(10, -15); //femtoseconds?
    //     float dtdt = pow(dt, 2);
    //     float nvt_bath = 0.0;
    //     torch::Tensor atom_types = output.at("atom_types").toTensor();
    //     // std::cout << "atom types: (" << atom_types.size(0) << ", " << atom_types.size(1) << ")" << std::endl;
    //     float atom_masses[nat];
    //     for (int i = 0; i < nat; i++) {
    //         int atom_type = atom_types[i][0].item<int>();
    //         if (atom_type == 0) {
    //             atom_masses[i] = 178.49;
    //         } else {
    //             atom_masses[i] = 15.999;
    //         }
    //     }
    //     float velocities[nat][3];
    //     memset(velocities, 0, nat*3*sizeof(float));
    //     float modified_acc[nat][3];
    //     for (int i = 0; i < nat; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             modified_acc[i][j] = forces_tensor[i][j].item<float>() / atom_masses[i] - nvt_bath * velocities[i][j];
    //             if (modified_acc[i][j] != modified_acc[i][j]) {
    //                 modified_acc[i][j] = 0;
    //             }
    //         }
    //     }
    //     float pos_fullstep[nat][3];
    //     for (int i = 0; i < nat; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             pos_fullstep[i][j] = new_pos[i][j].item<float>() + dt * velocities[i][j] + 0.5 * dtdt * modified_acc[i][j];
    //             if (pos_fullstep[i][j] != pos_fullstep[i][j]) {
    //                 pos_fullstep[i][j] = new_pos[i][j].item<float>();
    //             }
    //         }
    //     }
    //     float vel_halfstep[nat][3];
    //     for (int i = 0; i < nat; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             vel_halfstep[i][j] = velocities[i][j] + 0.5 * dt * modified_acc[i][j];
    //             if (vel_halfstep[i][j] != vel_halfstep[i][j]) {
    //                 vel_halfstep[i][j] = 0;
    //             }
    //         }
    //     }
    //     torch::Tensor pos_updated_after_physics = torch::from_blob(pos_fullstep, {nat, 3});
    //     float kB = 1.380649 * pow(10, -4) / 1.602176634;
    //     float e_kin_sum = 0;
    //     for (int i = 0; i < nat; i++) {
    //         float vel_sq_sum = 0;
    //         for (int j = 0; j < 3; j++) {
    //             vel_sq_sum += velocities[i][j] * velocities[i][j];
    //         }
    //         e_kin_sum += vel_sq_sum * atom_masses[i];
    //     }
    //     float e_kin_diff = 0.5 * (e_kin_sum - (3 * nat + 1) * kB * temp);
    //     float nvt_bath_halfstep = nvt_bath + 0.5 * dt * e_kin_diff / nvt_q;
    //     float e_kin_sum_halfstep = 0;
    //     for (int i = 0; i < nat; i++) {
    //         float vel_sq_sum_halfstep = 0;
    //         for (int j = 0; j < 3; j++) {
    //             vel_sq_sum_halfstep += vel_halfstep[i][j] * vel_halfstep[i][j];
    //         }
    //         e_kin_sum += vel_sq_sum_halfstep * atom_masses[i];
    //     }
    //     float e_kin_diff_halfstep = 0.5 * (e_kin_sum_halfstep - (3 * nat + 1) * kB * temp);
    //     nvt_bath = nvt_bath_halfstep + 0.5 * dt * e_kin_diff_halfstep / nvt_q;
    //     for (int i = 0; i < nat; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             velocities[i][j] = vel_halfstep[i][j] + 0.5 * dt * (forces_tensor[i][j].item<float>() / atom_masses[i]);
    //         }
    //     }


    //     std::vector<torch::jit::IValue> inputs2;
    //     torch::Dict<std::string, torch::Tensor> dictionary2;

    //     for (DictEntry *entry = arrays; entry; entry = entry->next) {
    //         int cols = 1;
    //         if (entry->ncols != 0) {
    //             cols = entry->ncols;
    //         }

    //         std::string str1 (entry->key);
    //         if (entry->data_t == data_s) {
    //             int arr[cols];
    //             for (int j=0; j<cols; j++) {
    //                 std::string type(((char**) (entry->data))[j]);
    //                 if (type == "Hf") {
    //                     arr[j] = 0;
    //                 } else if (type == "O") {
    //                     arr[j] = 1;
    //                 }
    //             }
    //             auto options = torch::TensorOptions().dtype(torch::kInt32);
    //             torch::Tensor t = torch::from_blob(arr, {cols, 1}, options=options).to(torch::kInt64);
    //             dictionary2.insert("atom_types", t);

    //             AtomGraph system(pos_updated_after_physics, cols, 5.0);
    //             torch::Tensor edges = system.edges();
    //             dictionary2.insert("edge_index", edges);
    //             std::cout << "step: " << step << ", Edge dim 0: " << edges.size(0) << ", dim 1: " << edges.size(1) << std::endl;
    //             dictionary2.insert("pos", pos_updated_after_physics);
    //         }
    //     }

    //     torch::Tensor t3_2 = torch::randint(0, 1, {nat}).to(torch::kLong);
    //     dictionary2.insert("batch", t3_2);
    //     inputs2.push_back(dictionary2);

    //     // Execute the model and turn its output into a tensor.
    //     // module.eval();
    //     std::cout << "step: " << step << ", before forward" << std::endl;
    //     auto output2 = module(inputs2).toGenericDict();
    //     // auto output2 = module.forward(inputs2).toGenericDict();
    //     std::cout << "step: " << step << ", after forward" << std::endl;

    //     // std::cout << "SECOND TIME keys\n--------------------------------------------------------" << std::endl;
    //     // for (auto it = output2.begin(); it!=output2.end(); it++) {
    //     //     std::cout << it->key() << std::endl;
    //     // }
    //     // std::cout << "SECOND TIME keysdone\n--------------------------------------------------------" << std::endl;
    //     // size: (1857, 3)
    //     torch::Tensor forces_tensor2 = output2.at("forces").toTensor();
    //     // std::cout << "SECOND TIME FORCES" << std::endl;
    //     // for (int i = 0; i < nat; i++) {
    //     //     std::cout << i << ": " << forces_tensor2[i][0] << ", " << forces_tensor2[i][1] << ", " << forces_tensor2[i][2] << std::endl;
    //     // }
    //     torch::Tensor total_energy_tensor2 = output2.at("total_energy").toTensor();
    //     torch::Tensor atomic_energy_tensor2 = output2.at("atomic_energy").toTensor();
    //     // std::cout << "SECOND TIME energy dim: " << atomic_energy_tensor2.dim() << std::endl;
    //     // std::cout << "SECOND TIME energy dim: " << atomic_energy_tensor2.size(0) << ", " << atomic_energy_tensor2.size(1) << std::endl;
    //     // std::cout << "SECOND TIME ENERGIES" << std::endl;
    //     // for (int i = 0; i < nat; i++) {
    //     //     std::cout << i << ": " << atomic_energy_tensor2[i][0] << std::endl;
    //     // }
    //     // auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
    //     float atomic_energy_sum2 = atomic_energy_tensor2.sum().data_ptr<float>()[0];
    //     std::cout << "step: " << step << ", Atomic energy sum: " << atomic_energy_sum2 << std::endl;
    //     torch::Tensor poses_final = output2.at("pos").toTensor();
    //     for (int i = 0; i < nat; i++) {
    //         std::cout << i << ": " << poses_final[i][0].item<float>() << ", " << poses_final[i][1].item<float>() << ", " << poses_final[i][2].item<float>() << std::endl;
    //     std::cout << i << ": " << atoms_answer[i * 3 + 0] << ", " << atoms_answer[i * 3 + 1] << ", " << atoms_answer[i * 3 + 2] << std::endl;
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////
