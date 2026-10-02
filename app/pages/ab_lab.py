"""A/B lab: the pipeline's experiment statistics, run with the visitor's own settings."""

from typing import Any

import streamlit as st

from recsys.abtest.simulation import SRM_ALPHA
from recsys.demo import charts, lab, ui
from recsys.demo.data import POLICY_ORDER

SEED = 0  # fixed, so the same settings always give the same simulated results
SIMULATIONS = (200, 500, 1000, 2000)
PEEKING_SIMULATIONS = 1000
POWER_OPTIONS = (0.7, 0.8, 0.9, 0.95)
ALPHA_OPTIONS = (0.01, 0.05, 0.10)
EXPERIMENT_SIZES = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def with_value(options: tuple[float, ...], value: float) -> list[float]:
    """The slider's usual options plus whatever the pipeline was configured with."""
    return sorted({*options, value})


def peeking_rates(result: dict[str, Any], alpha: float) -> dict[str, float]:
    return {
        f"stop at the first p < {alpha:g}": result["naive_rejection_rate"],
        "look once, at the end": result["fixed_horizon_rejection_rate"],
        "always-valid p (mSPRT)": result["msprt_rejection_rate"],
    }


def looks_control(most: int, configured: int) -> int | None:
    """How many times the simulated experiments are checked, within what the split allows."""
    if most < 2:
        st.warning(
            "Too few users in the smaller arm to simulate peeking at this split; move the share "
            "of users in treatment closer to 50%."
        )
        return None
    if most == 2:
        st.caption("At this split the smaller arm only supports 2 looks.")
        return 2
    if "lab_looks" in st.session_state:  # keep the visitor's choice, within the new limit
        st.session_state["lab_looks"] = min(st.session_state["lab_looks"], most)
        return st.slider("Looks during the experiment", 2, most, key="lab_looks")
    return st.slider("Looks during the experiment", 2, most, min(configured, most), key="lab_looks")


st.title("A/B lab")
st.markdown(
    "An **A/B test** splits users at random: *control* sees the current recommender, "
    "*treatment* the new one, and the difference in an outcome between the groups estimates the "
    "new one's effect. The outcome here is how many of a session's videos the user liked. These "
    "users reacted to almost every video, so the lab also knows the **true effect** and can show "
    "when a test gets it right and when it does not."
)
paths = ui.demo_paths()
status = ui.status(paths)
ready = [ui.require(status[name]) for name in ("ab_report", "ab_outcomes")]
if not all(ready):
    st.stop()
ab = ui.ab_report(paths.ab_report)
design = ab["design"]
outcomes = ui.stamp(paths.user_outcomes)
policies = [policy for policy in POLICY_ORDER if policy in ab["policies"]]

with st.container(border=True):
    control_column, treatment_column, share_column, alpha_column = st.columns(4)
    control = control_column.selectbox(
        "Control (current)", policies, index=default_index(policies, "als"), key="lab_control"
    )
    treatment = treatment_column.selectbox(
        "Treatment (new)",
        policies,
        index=default_index(policies, "two_tower"),
        key="lab_treatment",
    )
    share = share_column.slider(
        "Share of users in treatment",
        0.1,
        0.9,
        float(design["treatment_share"]),
        0.05,
        key="lab_share",
    )
    alpha = alpha_column.select_slider(
        "Significance level (alpha)",
        options=with_value(ALPHA_OPTIONS, design["alpha"]),
        value=design["alpha"],
        key="lab_alpha",
    )
level = charts.ci_label(1 - alpha)  # intervals match the test: 95% at alpha 0.05
if control == treatment:
    st.warning(
        "The same policy in both arms is an **A/A test**: the true effect is zero, so any "
        "'significant' result is a false alarm."
    )

run_tab, plan_tab, repeat_tab, peeking_tab, bug_tab = st.tabs(
    ["Run it", "Plan its size", "Repeat it", "Peeking", "Logging bug"]
)

with run_tab:
    st.markdown(
        "Users are split by a hash of their id and a **salt**. The same salt always gives the "
        "same split; a new salt gives a new random split of the same users."
    )
    base_salt = ab["assignment"]["salt"]
    if "lab_salt" not in st.session_state:
        st.session_state["lab_salt"] = base_salt
    if st.button("New random split", key="lab_reshuffle", icon=":material/shuffle:"):
        st.session_state["lab_splits"] = st.session_state.get("lab_splits", 0) + 1
        st.session_state["lab_salt"] = f"{base_salt}-{st.session_state['lab_splits']}"
    salt = st.text_input("Salt", key="lab_salt")
    result = ui.guarded(ui.experiment, *outcomes, control, treatment, salt, share, alpha)
    if result is not None:
        cuped, guard = result["cuped"], result["guardrail"]
        truth_card, estimate_card, p_card, decision_card = st.columns(4)
        truth_card.metric(
            "True effect",
            f"{result['true_effect']:+.3f}",
            help="Known here only because every user's reaction to every video is recorded",
        )
        estimate_card.metric(
            "Estimate (CUPED)",
            f"{cuped['difference']:+.3f}",
            help=f"{level} CI [{cuped['ci_low']:+.3f}, {cuped['ci_high']:+.3f}]",
        )
        p_card.metric("p-value", f"{cuped['p_value']:.3f}", help=f"significant below {alpha:g}")
        decision_card.metric("Decision", result["decision"])
        st.altair_chart(charts.readout_chart(result, 1 - alpha), width="stretch")
        st.caption(
            f"Lines are {level} confidence intervals (1 minus alpha). Grey: the plain difference "
            "in means. Blue: CUPED, which subtracts what each user's behaviour before the "
            "experiment predicts, for a narrower interval. Black: the truth."
        )
        with st.expander("The checks behind the decision"):
            st.markdown(
                f"- **Arm sizes**: {cuped['n_control']:,} control, {cuped['n_treatment']:,} "
                f"treatment; sample-ratio check p = {result['srm_p_value']:.3f} (the result is "
                f"invalid below {SRM_ALPHA:g}).\n"
                "- **Balance before the experiment**: p = "
                f"{result['covariate_balance_p_value']:.3f} for a difference in the arms' "
                "earlier like rates (a pre-period A/A test).\n"
                f"- **CUPED** removed {cuped['variance_reduction']:.0%} of the variance.\n"
                f"- **Guardrail**, seconds watched per session: {guard['difference']:+.1f} s, "
                f"{level} CI [{guard['ci_low']:+.1f}, {guard['ci_high']:+.1f}]. It blocks "
                "shipping only when treatment is significantly worse."
            )

with plan_tab:
    st.markdown(
        "Before running a test you ask how small an effect it could reliably detect: the "
        "**minimum detectable effect** (MDE). It shrinks with the square root of the number of "
        "users, so halving it takes four times as many."
    )
    sizes = sorted({*EXPERIMENT_SIZES, ab["users"]})
    size_column, power_column = st.columns(2)
    n_users = size_column.select_slider(
        "Users in the experiment (both arms)",
        options=sizes,
        value=ab["users"],
        format_func=lambda n: f"{n:,}",
        key="lab_users",
    )
    power = power_column.select_slider(
        "Power (chance of detecting a real effect)",
        options=with_value(POWER_OPTIONS, design["power"]),
        value=design["power"],
        format_func=lambda p: f"{p:.0%}",
        key="lab_power",
    )
    planned = ui.plan(*outcomes, control, treatment, n_users, share, alpha, power)
    needed = planned["users_needed"]
    mde_card, cuped_card, needed_card = st.columns(3)
    mde_card.metric(
        "Smallest detectable effect", f"{planned['mde']:.3f}", help="liked videos per session"
    )
    cuped_card.metric(
        "With CUPED",
        f"{planned['mde_cuped']:.3f}",
        help=f"pre-experiment covariate correlation {planned['covariate_correlation']:.2f}",
    )
    fewest, most = needed["range"]
    needed_card.metric(
        "Users the true effect needs",
        "none (no effect)" if needed["at_true_effect"] is None else f"{needed['at_true_effect']:,}",
        help=(
            None
            if fewest is None
            else f"{fewest:,} to {'unbounded' if most is None else f'{most:,}'} across the "
            f"{level} interval of the effect new users could show"
        ),
    )
    effects = {
        name: experiment["readout"]["true_effect"] for name, experiment in ab["experiments"].items()
    }
    planned_pairs = {(e["control"], e["treatment"]) for e in ab["experiments"].values()}
    if (control, treatment) not in planned_pairs:  # a pair the pipeline did not plan
        effects[f"{control} to {treatment} (chosen)"] = planned["true_effect"]
    curve = lab.mde_curve(planned["sd"], sizes, share, alpha, power)
    st.altair_chart(charts.mde_chart(curve, n_users, effects), width="stretch")
    st.caption(
        "Blue: the smallest effect each experiment size detects. Dashed: true effects of real "
        "comparisons. Black: the size chosen above. An effect below the curve there is likely "
        "to be missed."
    )

with repeat_tab:
    st.markdown(
        "One experiment is one random split. Repeating it with fresh splits shows how the method "
        "behaves: how often it detects the effect (**power**; with no effect, the **false-alarm "
        f"rate**), whether its estimates are right on average (**bias**), and whether its {level} "
        f"intervals contain the truth {level} of the time (**coverage**)."
    )
    n_sims = st.select_slider(
        "Repetitions",
        options=SIMULATIONS,
        value=1000,
        format_func=lambda n: f"{n:,}",
        key="lab_sims",
    )
    repeated = ui.guarded(ui.rerandomise, *outcomes, control, treatment, n_sims, share, alpha, SEED)
    if repeated is not None:
        rate_label = "False alarms" if control == treatment else "Power"
        plain_card, cuped_rate_card, bias_card, coverage_card = st.columns(4)
        plain_card.metric(f"{rate_label} (plain)", f"{repeated['rejection_rate']:.1%}")
        cuped_rate_card.metric(f"{rate_label} (CUPED)", f"{repeated['rejection_rate_cuped']:.1%}")
        bias_card.metric("Bias (CUPED)", f"{repeated['bias_cuped']:+.3f}")
        coverage_card.metric(f"{level} CI coverage (CUPED)", f"{repeated['ci_coverage_cuped']:.1%}")
        st.altair_chart(
            charts.histogram_chart(repeated["p_value_histogram_cuped"]), width="stretch"
        )
        st.caption(
            "With no real effect, p-values spread evenly (dashed line). A real effect piles "
            "them up near zero; the share below alpha is the power."
        )

with peeking_tab:
    st.markdown(
        "Checking a running test every day and stopping at the first p < alpha feels harmless, "
        "but every look is another chance for noise to cross the line. **Always-valid p-values** "
        "(mSPRT) may be checked at every look; the price is some power."
    )
    looks = looks_control(lab.max_looks(ab["users"], share), int(design["looks"]))
    if looks is not None:
        tau = ab["peeking"]["tau"]
        settings = (looks, PEEKING_SIMULATIONS, share, alpha, tau, SEED)
        aa = ui.guarded(ui.peek, *outcomes, control, control, *settings)
        real = (
            None
            if control == treatment
            else ui.guarded(ui.peek, *outcomes, control, treatment, *settings)
        )
        no_effect_column, effect_column = st.columns(2)
        if aa is not None:
            no_effect_column.markdown(
                f"**No real effect** ({control} in both arms): every 'significant' result is a "
                "false alarm."
            )
            no_effect_column.altair_chart(
                charts.rates_chart(peeking_rates(aa, alpha), alpha, "false-alarm rate"),
                width="stretch",
            )
        if real is not None:
            effect_column.markdown(
                f"**A real effect** ({control} to {treatment}, true effect "
                f"{real['true_effect']:+.2f}): how often it is detected."
            )
            effect_column.altair_chart(
                charts.rates_chart(peeking_rates(real, alpha), None, "share detected"),
                width="stretch",
            )
        st.caption(
            f"{PEEKING_SIMULATIONS:,} simulated experiments in which users arrive in random "
            f"order, checked {looks} times. Dashed: alpha = {alpha:g}, the false-alarm rate the "
            f"test promises. mSPRT is tuned to effects around {tau:.2f} (the planned MDE)."
        )

with bug_tab:
    st.markdown(
        "Suppose the new version only logs a session once the user likes something, so the "
        "least engaged treatment users vanish from the data. Both arms here run the same policy: "
        "the true effect is zero and any lift is fake. The **sample ratio mismatch** check "
        "compares the arm sizes with the intended split."
    )
    policy_column, drop_column = st.columns(2)
    policy = policy_column.selectbox(
        "Policy in both arms",
        policies,
        index=default_index(policies, "two_tower"),
        key="lab_bug_policy",
    )
    drop = drop_column.slider("Share of treatment users lost", 0.0, 0.5, 0.2, 0.05, key="lab_drop")
    bug = ui.guarded(
        ui.logging_bug, *outcomes, policy, ab["assignment"]["salt"], share, drop, alpha
    )
    if bug is not None:
        with_bug, without_bug = bug["with_bug"], bug["without_bug"]
        fake_lift = with_bug["estimate"]["difference"] - without_bug["estimate"]["difference"]
        estimate_card, srm_card, bug_decision_card = st.columns(3)
        estimate_card.metric(
            "Estimate with the bug",
            f"{with_bug['estimate']['difference']:+.3f}",
            delta=f"{fake_lift:+.3f} fake lift",
            delta_color="off",
            help=f"{with_bug['users_dropped']:,} treatment users lost; without the bug "
            f"{without_bug['estimate']['difference']:+.3f}",
        )
        srm_card.metric(
            "Sample-ratio p-value",
            f"{with_bug['srm_p_value']:.2g}",
            help=f"alarm below {SRM_ALPHA:g}",
        )
        bug_decision_card.metric("Decision with the bug", with_bug["decision"])
        st.caption(
            "Small losses fake small lifts and pass the check; the check is a test too, with "
            "limited power. Large losses trip it and the result is declared invalid."
        )
