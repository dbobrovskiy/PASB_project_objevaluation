import numpy as np
import pandas as pd
import cobra
import cplex
from cobra.io import read_sbml_model, load_matlab_model
from cobra.flux_analysis import flux_variability_analysis

def get_producing_reactions(model, metabolite):
    """
    Returns a list of reactions that produce the given metabolite
    (i.e. the metabolite is a product of the reaction or it's a substrate of a reversible reaction)
    """
    producing_reactions = []
    for reaction in model.reactions:
        if metabolite in [metab.id for metab in reaction.products]:
            producing_reactions.append(reaction)
        elif metabolite in [metab.id for metab in reaction.reactants] and reaction.reversibility:
            producing_reactions.append(reaction)
    return producing_reactions

def get_consuming_reactions(model, metabolite):
    """
    Returns a list of reactions that consume the given metabolite
    (i.e. the metabolite is a substrate of the reaction or it's a product of a reversible reaction)
    """
    consuming_reactions = []
    for reaction in model.reactions:
        if metabolite in [metab.id for metab in reaction.reactants]:
            consuming_reactions.append(reaction)
        elif metabolite in [metab.id for metab in reaction.products] and reaction.reversibility:
            consuming_reactions.append(reaction)
    return consuming_reactions

def calculate_split_ratio(model, reaction, metabolite):
    """
    Computes the split ratio of a given reaction.
    That is, the flux of the reaction multiplied by the stoichiometric coefficient of the metabolite
    and divided by the sum of the fluxes of reactions that produce the metabolite
    (also multiplied by the stoichiometric coefficient of the metabolite).
    """
    reaction = model.reactions.get_by_id(reaction)
    if reaction.flux == 0:
        return 0
    split_ratio = 0
    ratio_denom = 0
    split_ratio = -1 * reaction.flux * reaction.get_coefficient(metabolite)
    if reaction.flux < 0:
        cons_reactions = get_consuming_reactions(model, metabolite)
        for cons_reaction in cons_reactions:
            if cons_reaction.id == reaction.id:
                continue
            ratio_denom += cons_reaction.flux * cons_reaction.get_coefficient(metabolite)
    else:
        prod_reactions = get_producing_reactions(model, metabolite)
        for prod_reaction in prod_reactions:
            if prod_reaction.id == reaction.id:
                continue
            ratio_denom += prod_reaction.flux * prod_reaction.get_coefficient(metabolite)
    split_ratio /= ratio_denom
    return split_ratio

def analyze_flux_variability(
    model: cobra.Model,
    fluxes: pd.DataFrame,
    total: bool = True,
) -> pd.DataFrame:
    """
    For reactions in the fluxes DataFrame, computes the split ratios and the flux variability
    (minimum and maximum fluxes) for the given model.

    Parameters
    ----------
        model: cobra.Model
            The model to analyze.
        fluxes: pd.DataFrame
            A DataFrame containing the fluxes to compute the split ratios for.
        total: bool
            Whether to compute the split ratios for all of the reactions in the fluxes DataFrame
            or only for Shuetz reactions.
    """
    # retrieve the reactions to compute the split ratios for
    if total:
        reactions = fluxes["iAF1260"]
    else:
        reactions = fluxes["iAF1260"][fluxes["Schuetz"].str.startswith("R")]

    # run variability analysis
    results = flux_variability_analysis(model, [model.reactions.get_by_id(reaction) for reaction in reactions])

    # compute split ratios
    split_ratios = []
    for i, reaction in enumerate(results.index.values):
        metabolite = fluxes["Metab_norm"][fluxes["iAF1260"] == reaction].values[0]
        split_ratios.append(calculate_split_ratio(model, reaction, metabolite))
    results["split_ratio"] = split_ratios

    return results

def compute_predictive_fidelity(
    model: cobra.Model,
    fluxes: pd.DataFrame,
    fba_results: pd.DataFrame,
    total: bool = True,
    condition: str = None,
) -> float:
    """
    Computes the predictive fidelity for the given model and fluxes.

    Parameters
    ----------
        model: cobra.Model
            The model to compute the predictive fidelity for.
        fluxes: pd.DataFrame
            A DataFrame containing the fluxes to compute the predictive fidelity for.
        fba_results: pd.DataFrame
            A DataFrame containing the FBA results for the model.
        total: bool
            Whether to compute the predictive fidelity for all of the reactions in the fluxes DataFrame
            or only for Shuetz reactions.
        condition: str
            Name of the fluxes column containing the experimental fluxes.
    """
    # retrieve the reactions to compute the predictive fidelity for
    if not total:
        fluxes = fluxes[fluxes["Schuetz"].str.startswith("R")]
        fluxes.index = range(fluxes.shape[0])
    reactions = fluxes["iAF1260"]
    
    # transform fluxes to a DataFrame of mean and std
    exp_fluxes = fluxes[condition].str.split(" ± ", expand=True).astype(float) / 100
    exp_fluxes.columns = ["mean", "std"]
    exp_fluxes["mean"] = exp_fluxes["mean"] * fluxes["iAF_sign"]

    # compute the predictive fidelity
    diff = [fba_results.fluxes[reaction] - exp_fluxes["mean"][i] for i, reaction in enumerate(reactions)]
    weights = [1 / exp_fluxes["std"][i] for i in range(exp_fluxes.shape[0])]
    weights = [weight / sum(weights) for weight in weights]
    predictive_fidelity = sum([weight*diff[i]**2 for i, weight in enumerate(weights)])

    return predictive_fidelity

def set_fidelity_objective(
    model: cobra.Model,
    fluxes: pd.DataFrame,
    fba_results: pd.DataFrame,
    total: bool = True,
    condition: str = None,
    direction: str = "min",
    constraint: str = None,
) -> cobra.Model:
    """
    Sets the objective of the model to the predictive fidelity.

    Parameters
    ----------
        model: cobra.Model
            The model to set the objective for.
        fluxes: pd.DataFrame
            A DataFrame containing the fluxes to compute the predictive fidelity for.
        fba_results: pd.DataFrame
            A DataFrame containing the FBA results for the model.
        total: bool
            Whether to compute the predictive fidelity for all of the reactions in the fluxes DataFrame
            or only for Shuetz reactions.
        condition: str
            Name of the fluxes column containing the experimental fluxes.
        direction: str
            Either "max" or "min": whether to maximize or minimize the predictive fidelity.
        constraint: str
            Which flux to keep constant (reaction ID).
    """
    # retrieve the reactions to compute the predictive fidelity for
    if not total:
        fluxes = fluxes[fluxes["Schuetz"].str.startswith("R")]
        fluxes.index = range(fluxes.shape[0])
    reactions = fluxes["iAF1260"]
    
    # transform fluxes to a DataFrame of mean and std
    exp_fluxes = fluxes[condition].str.split(" ± ", expand=True).astype(float) / 100
    exp_fluxes.columns = ["mean", "std"]
    exp_fluxes["mean"] = exp_fluxes["mean"] * fluxes["iAF_sign"]

    # create a constraint to keep the flux of the given reaction constant
    cons = model.problem.Constraint(
        model.reactions.get_by_id(constraint).flux_expression,
        lb=fba_results.fluxes[constraint],
        ub=fba_results.fluxes[constraint]
    )
    model.add_cons_vars(cons)

    # create an objective
    diff = [model.reactions.get_by_id(reaction).flux_expression - exp_fluxes["mean"][i] for i, reaction in enumerate(reactions)]
    weights = [1 / exp_fluxes["std"][i] for i in range(exp_fluxes.shape[0])]
    weights = [weight / sum(weights) for weight in weights]
    objective = model.problem.Objective(
        sum([weight*diff[i]**2 for i, weight in enumerate(weights)]),
        direction=direction
    )
    model.objective = objective

    return model

def load_model():
    """
    Loads the iAF1260 model from the SBML file.
    """
    model = read_sbml_model("iAF1260.xml")

    # iterate through all reactions; if the bound is less than 0 and higher than -999999, set it to -999999
    # if the bound is higher than 0 and lower than 999999, set it to 999999
    for reaction in model.reactions:
        if reaction.lower_bound < 0 and reaction.lower_bound > -999999:
            reaction.lower_bound = -999999
        elif reaction.lower_bound > 0 and reaction.lower_bound < 999999:
            reaction.lower_bound = 0
        if reaction.upper_bound < 0 and reaction.upper_bound > -999999:
            reaction.upper_bound = 0
        elif reaction.upper_bound > 0 and reaction.upper_bound < 999999:
            reaction.upper_bound = 999999

    # set glucose uptake to -1
    model.reactions.get_by_id("EX_glc__D_e").lower_bound = -1

    return model

def evaluate_objective(
    model: cobra.Model,
    fluxes: pd.DataFrame,
    objective: str = "biomass",
) -> pd.DataFrame:
    """
    For a given objective:
        1. Optimize the model.
        2. Perform flux variability analysis.
        3. If fluxes aren't variable, compute predictive fidelity.
        4. If fluxes are variable:
            a. Maximize and minimize predictive fidelity.
            b. Run parsimonious FBA.
            c. Run geometric FBA.
    
    Parameters
    ----------
    model : cobra.Model
        The model to evaluate.
    fluxes : pd.DataFrame
        A DataFrame containing the fluxes to compute the predictive fidelity for.
    objective : str
        The objective to set for the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the evaluation.
        Columns: "objective", "method", "reaction_set", "Batch_06", "Batch_03", "Batch_02", "Chem_01", "Chem_04C", "Chem_04N"
    """
    # set the objective
    model.solver = "cplex"
    if objective == "biomass":
        model.objective = model.reactions.get_by_id("BIOMASS_Ec_iAF1260_core_59p81M")
    elif objective == "ATP":
        model.objective = model.reactions.get_by_id("ATPM")
    elif objective == "minfluxsq":
        model.objective = model.problem.Objective(
            sum([reaction.flux_expression**2 for reaction in model.reactions]),
            direction="min"
        )
    elif objective == "minfluxabs":
        model.objective = model.problem.Objective(
            sum([abs(reaction.flux_expression) for reaction in model.reactions]),
            direction="min"
        )
    elif objective == "ATPperflux":
        model.objective = model.problem.Objective(
            model.reactions.get_by_id("ATPM").flux_expression / sum([(reaction.flux_expression)**2 for reaction in model.reactions]),
            direction="max"
        )
    elif objective == "biomassperflux":
        model.objective = model.problem.Objective(
            model.reactions.get_by_id("BIOMASS_Ec_iAF1260_core_59p81M").flux_expression / sum([abs(reaction.flux_expression) for reaction in model.reactions]),
            direction="max"
        )
    else:
        raise ValueError("Invalid objective.")
    model_cp = model.copy()
    
    solution = model.optimize()
    conditions = ["Batch_06", "Batch_03", "Batch_02", "Chem_01", "Chem_04C", "Chem_04N"]

    # if the objective is linear, account for variability in fluxes
    if objective in ["biomass", "ATP"]:
        print("Linear objective, running flux variability analysis...")
        
        # run flux variability analysis
        fva = flux_variability_analysis(model, model.reactions)

        # if fluxes aren't variable, compute predictive fidelity
        if np.isclose(fva["maximum"], fva["minimum"]).all():
            print("All fluxes are constant.")
            fidelities = []
            for reaction_set in ["total", "shuetz"]:
                for condition in conditions:
                    fidelities.append(compute_predictive_fidelity(model, fluxes, solution,
                                                                  total=(reaction_set=="total"), condition=condition))
            results = pd.DataFrame(
                {
                    "objective": [objective] * 2,
                    "method": ["fba"] * 2,
                    "reaction_set": ["total", "shuetz"],
                    "Batch_06": [fidelities[0], fidelities[6]],
                    "Batch_03": [fidelities[1], fidelities[7]],
                    "Batch_02": [fidelities[2], fidelities[8]],
                    "Chem_01": [fidelities[3], fidelities[9]],
                    "Chem_04C": [fidelities[4], fidelities[10]],
                    "Chem_04N": [fidelities[5], fidelities[11]]
                }
            )
            return results

        print("Fluxes are variable.")

        # maximize/minimize predictive fidelity
        print("Maximizing and minimizing predictive fidelity...")
        fidmax = []
        fidmin = []
        for reaction_set in ["total", "shuetz"]:
            for condition in conditions:
                constraint = "BIOMASS_Ec_iAF1260_core_59p81M" if objective == "biomass" else "ATPM"
                # maximization of fidelity - will usually give an error, since the objective is not convex
                try:
                    model = set_fidelity_objective(model, fluxes, solution, total=(reaction_set=="total"),
                                                   condition=condition, direction="max", constraint=constraint)
                    model.optimize()
                    fidmax.append(model.objective.value)
                except:
                    fidmax.append(np.nan)
                # minimization of fidelity
                model = set_fidelity_objective(model, fluxes, solution, total=(reaction_set=="total"),
                                               condition=condition, direction="min", constraint=constraint)
                model.optimize()
                fidmin.append(model.objective.value)
    
        # run parsimonious FBA
        model = model_cp.copy()
        print("Running parsimonious FBA...")
        solution = cobra.flux_analysis.pfba(model)
        pfba_fidelities = []
        for reaction_set in ["total", "shuetz"]:
            for condition in conditions:
                pfba_fidelities.append(compute_predictive_fidelity(model, fluxes, solution,
                                                                   total=(reaction_set=="total"), condition=condition))
    
        # run geometric FBA - not working, results in cplex error
        #model = model_cp.copy()
        #print("Running geometric FBA...")
        #solution = cobra.flux_analysis.geometric_fba(model)
        #gfba_fidelities = []
        #for condition in conditions:
        #    gfba_fidelities.append(compute_predictive_fidelity(model, fluxes, solution, condition=condition))
        
        # create a DataFrame with the results
        results = pd.DataFrame(
            {
                "objective": [objective] * 6,
                "method": ["fidmax"]*2 + ["fidmin"]*2 + ["pfba"]*2,
                "reaction_set": ["total", "shuetz"] * 3,
                "Batch_06": [fidmax[0], fidmax[6], fidmin[0], fidmin[6], pfba_fidelities[0], pfba_fidelities[6]],
                "Batch_03": [fidmax[1], fidmax[7], fidmin[1], fidmin[7], pfba_fidelities[1], pfba_fidelities[7]],
                "Batch_02": [fidmax[2], fidmax[8], fidmin[2], fidmin[8], pfba_fidelities[2], pfba_fidelities[8]],
                "Chem_01": [fidmax[3], fidmax[9], fidmin[3], fidmin[9], pfba_fidelities[3], pfba_fidelities[9]],
                "Chem_04C": [fidmax[4], fidmax[10], fidmin[4], fidmin[10], pfba_fidelities[4], pfba_fidelities[10]],
                "Chem_04N": [fidmax[5], fidmax[11], fidmin[5], fidmin[11], pfba_fidelities[5], pfba_fidelities[11]]
            }
        )

    else:
        print("Quadratic objective, so fluxes aren't variable.")
        fidelities = []
        for reaction_set in ["total", "shuetz"]:
            for condition in conditions:
                fidelities.append(compute_predictive_fidelity(model, fluxes, solution,
                                                              total=(reaction_set=="total"), condition=condition))
        
        results = pd.DataFrame(
            {
                "objective": [objective] * 2,
                "method": ["fba"] * 2,
                "reaction_set": ["total", "shuetz"],
                "Batch_06": [fidelities[0], fidelities[6]],
                "Batch_03": [fidelities[1], fidelities[7]],
                "Batch_02": [fidelities[2], fidelities[8]],
                "Chem_01": [fidelities[3], fidelities[9]],
                "Chem_04C": [fidelities[4], fidelities[10]],
                "Chem_04N": [fidelities[5], fidelities[11]]
            }
        )

    return results