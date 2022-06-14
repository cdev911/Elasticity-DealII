/* ===================================
 * Nonlinear Elastic BVP (Neo-Hookean)
 * ===================================
 * Problem description:
 *   Nonlinear elastostastic solver for Neo-Hookean material model.
 *   The residual vector and tangent matrix is computed using Automatic differentation
 *
 *   Author: Chaitanya Dev
 *           Friedrich-Alexander University Erlangen-Nuremberg
 *
 *  References:
 *  Mergheim, Julia. Lecture Notes - NONLINEAR FINITE ELEMENT METHODS
 *  Deal.II 9.3 - Step 72
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/integrators/elasticity.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/elasticity/kinematics.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/algorithms/general_data_storage.h>
#include <deal.II/differentiation/ad.h>
#include <deal.II/base/timer.h>
// This again is C++:
#include <fstream>
#include <iostream>


namespace Elasticity_AD
{
  using namespace dealii;

 class Errors {
    /**
     * @brief This class is to compute the relative error, which is used in Newton Raphson scheme. 
     * The class is Initialized with error in the first iteration, subsequently in later iterations given an error value, it can return
     * normalized error wrt the error in first iteration.
     */
  private:
    double error_first_iter = 0.0;
    bool initialized = false;
  public:
  /**
   * @brief Initialize error.
   * 
   * @param error 
   */
    inline void Initialize(double error) {
      if (error == 0.0)
        throw std::runtime_error ("First iteration error cannot be 0.0 ");
      else {
        if (!initialized) {
          error_first_iter = error;
          initialized = true;
        } else
          std::cerr << "Already the error is initialized." << std::endl;
      }
    }

    /**
     * @brief Function to get the Normalized Error. 
     * 
     * @param error 
     * @return double 
     */
    inline double GetNormalizedError(double error) {
      if (initialized)
        return error / error_first_iter;
      else {
        std::cerr << "First iteration error not initialized, so cannot Normalize." << std::endl;
        return 1e9;
      }
    }

    /**
     * @brief Reset the error.
     * 
     */
    inline void Reset() {
      error_first_iter = 0.0;
      initialized = false;
    }
  };

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  class NeoHookElasticProblem
  {
  public:
    NeoHookElasticProblem();
    void run();

  private:
    void setup_system();
    void setup_constraint();
    void assemble_system();
    void solve(Vector<double> &newton_update);
    void solve_load_step_NR();
    void output_results(const unsigned int cycle) const;
    Vector<double> get_total_solution(const Vector<double> &solution_delta) const;
    double get_error_residual();

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    Vector<double> tria_boundary_ids;

    FESystem<dim> fe;

    const double lambda, mu;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> solution_delta;
    Vector<double> newton_update;
    Vector<double> residual;

    unsigned int newton_iteration = 0;
    unsigned int max_nr_steps = 20;
    double curr_load = 1.0;
    double accum_load = 0.0;
    double init_load = 1.0;
    int load_step;

    int terminate_loadstep = 0;

    Errors error_NR;

    mutable TimerOutput compute_timer;
  };



  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  NeoHookElasticProblem<dim, ADTypeCode>::NeoHookElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , lambda(1.0)
    , mu(0.5)
    , compute_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)

  {}

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::setup_system()
  {
    TimerOutput::Scope t(compute_timer, "setup_system");

    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    solution_delta.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    residual.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    tria_boundary_ids.reinit(triangulation.n_active_cells());
    Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
    unsigned int counter = 0, cell_counter = 0;
    for (auto cell: triangulation.active_cell_iterators()) {
      if (cell->at_boundary())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
          if (cell->face(f)->at_boundary()) {
            int boundary_id = cell->face(f)->boundary_id();
            if (dom_boundary_ids[counter] == 0) {
              dom_boundary_ids[counter] = boundary_id;
              tria_boundary_ids[cell_counter] = boundary_id;
            }
          }
          ++counter;
        }
      ++cell_counter;
    }

  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::setup_constraint()
  {
    TimerOutput::Scope t(compute_timer, "setup_constraint");

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const bool apply_dirichlet_bc = (newton_iteration == 0);

    if (apply_dirichlet_bc) {
      VectorTools::interpolate_boundary_values(dof_handler,
                                              2,
                                              Functions::ZeroFunction<dim>(dim),
                                              constraints);

      double step_fraction = curr_load;
      double current_load = 0.5 * step_fraction;

      std::vector<bool> comp_vec(dim, true);
      ComponentMask comp_mask(comp_vec);
      VectorTools::interpolate_boundary_values(dof_handler,
                                              3,
                                              Functions::ConstantFunction<dim>(current_load,dim),
                                              constraints,
                                              comp_mask);
    } else {
      VectorTools::interpolate_boundary_values(dof_handler,
                                        2,
                                        Functions::ZeroFunction<dim>(dim),
                                        constraints);

      std::vector<bool> comp_vec(dim, true);
      ComponentMask comp_mask(comp_vec);
      VectorTools::interpolate_boundary_values(dof_handler,
                                              3,
                                              Functions::ZeroFunction<dim>(dim),
                                              constraints,
                                              comp_mask);
    }
                                             
    constraints.close();
  }


  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::assemble_system()
  {
    TimerOutput::Scope t(compute_timer, "assemble_system");

    using ADHelper = Differentiation::AD::EnergyFunctional<Differentiation::AD::NumberTypes::sacado_dfad_dfad,double>;
    using ADNumberType = typename ADHelper::ad_type;

    Vector<double> current_solution = get_total_solution(solution_delta);

    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector u_fe(0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);
        const unsigned int n_independent_variables = dofs_per_cell;
        ADHelper           ad_helper(n_independent_variables);

        ad_helper.register_dof_values(current_solution, local_dof_indices);
        const std::vector<ADNumberType> &dof_values_ad =
          ad_helper.get_sensitive_dof_values();
        std::vector<Tensor<2, dim, ADNumberType>> old_solution_gradients(
          fe_values.n_quadrature_points);
        fe_values[u_fe].get_function_gradients_from_local_dof_values(
          dof_values_ad, old_solution_gradients);

        ADNumberType energy_ad = ADNumberType(0.0);
        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const ADNumberType lambda = this->lambda;
          const ADNumberType mu = this->mu;
          Tensor<2, dim, ADNumberType> F = Physics::Elasticity::Kinematics::F(old_solution_gradients[q]);
          SymmetricTensor< 2, dim, ADNumberType> C = Physics::Elasticity::Kinematics::C(F);
          
          const ADNumberType J = determinant(F);

          if (J <= 0){
            terminate_loadstep += 1;
            break;
          }

          const ADNumberType psi = mu*0.5*(trace(C) - 3)
                                - mu*std::log(J)
                                + lambda*0.5*std::pow(std::log(J),2.0);
          energy_ad += psi * fe_values.JxW(q);

        }
        ad_helper.register_energy_functional(energy_ad);
        ad_helper.compute_residual(cell_rhs);
        cell_rhs *= -1.0;
        ad_helper.compute_linearization(cell_matrix);

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, residual);
      }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::solve_load_step_NR()
  {

    error_NR.Reset();

    newton_iteration = 0;
    for (; newton_iteration < max_nr_steps; ++newton_iteration) {
      // Reset the tangent matrix and the rhs vector
      system_matrix = 0.0;
      residual = 0.0;

      setup_constraint();
      assemble_system();
      
      if(terminate_loadstep >= 1) {
        return;
      }

      if (newton_iteration == 0)
        error_NR.Initialize(get_error_residual());

      double error_residual_norm = error_NR.GetNormalizedError(get_error_residual());

      std::cout << error_residual_norm << " | ";

      /*Problem has to be solved at least once*/
      if (newton_iteration > 0 && error_residual_norm <= 1e-8)
        break;

      solve(newton_update);

      //ADD THE NEWTON INCREMENT TO THE LOAD STEP DELTA solution_delta
      solution_delta += newton_update;
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::solve(Vector<double> &newton_update_)
  {
    TimerOutput::Scope t(compute_timer, "solve");

    newton_update_ = 0.0;

    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, newton_update_, residual, preconditioner);

    constraints.distribute(newton_update_);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  Vector<double> 
  NeoHookElasticProblem<dim, ADTypeCode>::get_total_solution(const Vector<double> &solution_delta) const
  {
    Vector<double>  solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double 
  NeoHookElasticProblem<dim, ADTypeCode>::get_error_residual()
  {
    double res = 0.0;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
      if (!constraints.is_constrained(i)) {
        res += std::pow(residual(i),2.0);
      }
    }

    return res;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::output_results(const unsigned int cycle) const
  {
    TimerOutput::Scope t(compute_timer, "output_results");

    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);

    std::vector<std::string> boundary_names(1, "boundary_id"); 

    std::vector<std::string> solution_names(dim, "displacement");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
        scalar_data_component_interpretation(DataComponentInterpretation::component_is_scalar);

    if(cycle != 0){
        data_out.add_data_vector(dof_handler,
                                solution,
                                solution_names,
                                vector_data_component_interpretation);
    }
    data_out.add_data_vector(tria_boundary_ids,
                            boundary_names);
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void NeoHookElasticProblem<dim, ADTypeCode>::run()
  {
    const unsigned int number_load_steps = 5;
    init_load = 1.0 / number_load_steps;

    accum_load = 0.0;
    load_step = 0;
    curr_load = init_load;

    GridGenerator::hyper_cube(triangulation, -1, 1, true);
    triangulation.refine_global(4);

    setup_system();
    output_results(load_step);
    ++load_step;

    while (true) {

      solution_delta = 0.0;

      std::cout << "LS: " << load_step << " : ";
      solve_load_step_NR();
      std::cout << std::endl;
      if (newton_iteration >= max_nr_steps || terminate_loadstep >= 1) {
        curr_load *= 0.5;
        terminate_loadstep = 0;
        continue;
      }
      accum_load += curr_load;

      solution += solution_delta;

      output_results(load_step);
      ++load_step;

      if(accum_load >= 1.0)
        break;

      if (newton_iteration <= 5)
        curr_load *= 2.;

      init_load = curr_load;
      if(init_load > 1.)
        init_load = 1.;

      if (curr_load + accum_load > 1.)
        curr_load = 1. - accum_load;
    } // end loop over time steps

  }
} // namespace Elasticity_AD


int main()
{
  try
    {
      using namespace dealii;

      constexpr Differentiation::AD::NumberTypes ADTypeCode =
          Differentiation::AD::NumberTypes::sacado_dfad_dfad;
      Elasticity_AD::NeoHookElasticProblem<2, ADTypeCode> elastic_problem_2d;
      elastic_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
