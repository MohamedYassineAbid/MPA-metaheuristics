"""
Marine Predators Algorithm (MPA) - Streamlit Application
Interactive visualization and analysis tool
"""
import streamlit as st
import os
import sys
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import time

# Add cec2017-py to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cec2017-py'))

try:
    from cec2017.functions import all_functions
    CEC2017_AVAILABLE = True
except ImportError:
    CEC2017_AVAILABLE = False
    st.warning("CEC2017 functions not available. Please install the cec2017-py package.")

# Import MPA optimizer
from mpa_optimizer import mpa, mpa_with_callback

# Page configuration
st.set_page_config(
    page_title="MPA - Marine Predators Algorithm",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .stCodeBlock {
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("## 🐋 Marine Predators Algorithm")
    st.markdown("---")
    
    section = st.radio(
        "📍 Navigate to:",
        ["About", "Plots", "References"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>
        <p>MPA v1.0</p>
        <p>Built with Streamlit 🎈</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown('<div class="main-header">🐋 Marine Predators Algorithm</div>', 
            unsafe_allow_html=True)

# ============================================================================
# SECTION 1: ABOUT
# ============================================================================
if section == "About":
    st.markdown("## 📖 About the Algorithm")
    
    st.markdown("""
    ### Overview
    
    The **Marine Predators Algorithm (MPA)** is a nature-inspired metaheuristic optimization 
    algorithm that mimics the foraging behavior and interactions between predators and prey 
    in marine ecosystems. It was proposed by Faramarzi et al. in 2020.
    
    ### Key Concepts
    
    The algorithm is based on the widespread foraging strategy known as **Lévy** and **Brownian** movements, 
    which are optimal searching strategies observed in nature. MPA divides the optimization process into 
    three distinct phases:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Phase 1: Exploration**  
        🌊 High-velocity ratio  
        (Iter < Tmax/3)  
        Prey moves faster than predator
        """)
    
    with col2:
        st.warning("""
        **Phase 2: Transition**  
        ⚖️ Unit-velocity ratio  
        (Tmax/3 ≤ Iter < 2Tmax/3)  
        Equal velocities
        """)
    
    with col3:
        st.success("""
        **Phase 3: Exploitation**  
        🎯 Low-velocity ratio  
        (Iter ≥ 2Tmax/3)  
        Predator moves faster
        """)
    
    st.markdown("---")
    
    # Algorithm Description
    st.markdown("### 🔬 Algorithm Description")
    
    st.markdown("""
    **Main Components:**
    
    1. **Elite (Top Predator)**: The best solution found so far
    2. **Prey Population**: Search agents exploring the solution space
    3. **Marine Memory**: Mechanism to preserve better solutions
    4. **FADs Effect**: Fish Aggregating Devices that introduce diversity
    5. **Lévy Flight**: Long-range exploration strategy
    6. **Brownian Motion**: Local search mechanism
    7. **CF (Control Factor)**: Adaptive parameter balancing exploration/exploitation
    
    **Key Parameters:**
    - `N`: Number of search agents (prey)
    - `D`: Problem dimension
    - `Tmax`: Maximum iterations
    - `FADs`: Fish Aggregating Devices probability (default: 0.2)
    - `P`: Constant parameter (default: 0.5)
    - `β`: Lévy flight parameter (default: 1.5)
    """)
    
    st.markdown("---")
    
    # Pseudocode
    st.markdown("### 📋 Pseudocode")
    
    st.markdown("#### Lévy Flight Vector Generation")
    
    st.markdown("**Function:** LevyFlightVector$(N, D, \\beta=1.5)$")
    st.markdown("*Generate Levy flight matrix for N agents with D dimensions*")
    
    st.latex(r"\text{Calculate } \sigma_u \text{ using Gamma function:}")
    st.latex(r"\sigma_u \gets \left(\frac{\Gamma(1 + \beta) \cdot \sin(\pi\beta / 2)}{\Gamma((1 + \beta)/2) \cdot \beta \cdot 2^{(\beta-1)/2}}\right)^{1/\beta}")
    
    st.markdown("Generate random vectors from normal distributions:")
    st.latex(r"u \sim \mathcal{N}(0, \sigma_u), \quad \text{size: } (N, D)")
    st.latex(r"v \sim \mathcal{N}(0, 1), \quad \text{size: } (N, D)")
    
    st.markdown("Calculate step size:")
    st.latex(r"\text{step} \gets \frac{u}{|v|^{1/\beta}}")
    
    st.markdown("**Return:** $\\text{step}$")
    
    st.markdown("---")
    
    st.markdown("#### Main MPA Algorithm")
    
    st.markdown("**Function:** MPA$(f, D, N, LB, UB, T_{\\max}, FADs=0.2, P=0.5)$")
    
    # Initialization
    st.markdown("##### 1. Initialization")
    st.latex(r"\text{Prey} \gets \text{random\_uniform}(LB, UB, \text{size}=(N, D))")
    st.latex(r"X_{\min} \gets \text{matrix}(N, D) \text{ filled with } LB")
    st.latex(r"X_{\max} \gets \text{matrix}(N, D) \text{ filled with } UB")
    st.latex(r"\text{Top\_predator\_pos} \gets \mathbf{0}_D, \quad \text{Top\_predator\_fit} \gets \infty")
    st.latex(r"\text{fitness} \gets [\infty]_N")
    st.latex(r"\text{Prey\_old} \gets \text{Prey}, \quad \text{fit\_old} \gets \text{fitness}")
    
    # Main Loop
    st.markdown("##### 2. Main Optimization Loop")
    st.markdown("**For** $\\text{Iter} = 0$ **to** $T_{\\max}-1$ **do:**")
    
    st.markdown("**Phase 1: Detect Top Predator**")
    st.markdown("**For** $i = 0$ **to** $N-1$ **do:**")
    st.latex(r"\text{Prey}[i] \gets \text{clip}(\text{Prey}[i], LB, UB)")
    st.latex(r"\text{fitness}[i] \gets f(\text{Prey}[i])")
    st.markdown("**If** $\\text{fitness}[i] < \\text{Top\\_predator\\_fit}$ **then:**")
    st.latex(r"\text{Top\_predator\_fit} \gets \text{fitness}[i]")
    st.latex(r"\text{Top\_predator\_pos} \gets \text{Prey}[i]")
    
    st.markdown("**Marine Memory Saving**")
    st.markdown("**If** $\\text{Iter} = 0$ **then:**")
    st.latex(r"\text{fit\_old} \gets \text{fitness}, \quad \text{Prey\_old} \gets \text{Prey}")
    st.latex(r"\text{Inx} \gets (\text{fit\_old} < \text{fitness})")
    st.latex(r"\text{Prey} \gets \text{where}(\text{Inx}, \text{Prey\_old}, \text{Prey})")
    st.latex(r"\text{fitness} \gets \text{where}(\text{Inx}, \text{fit\_old}, \text{fitness})")
    st.latex(r"\text{fit\_old} \gets \text{fitness}, \quad \text{Prey\_old} \gets \text{Prey}")
    
    # Movement Strategy
    st.markdown("##### 3. Movement Strategy")
    st.markdown("**Elite Matrix & Control Factor:**")
    st.latex(r"\text{Elite} \gets \text{replicate}(\text{Top\_predator\_pos}, N)")
    st.latex(r"CF \gets \left(1 - \frac{\text{Iter}}{T_{\max}}\right)^{2 \cdot \text{Iter}/T_{\max}}")
    
    st.markdown("**Generate Movement Vectors:**")
    st.latex(r"RL \gets 0.05 \times \text{LevyFlightVector}(N, D, \beta=1.5)")
    st.latex(r"RB \sim \mathcal{N}(0, 1), \quad \text{size: } (N, D)")
    
    st.markdown("**Update Prey Positions:**")
    st.markdown("**For** $i = 0$ **to** $N-1$ **do:**")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;**For** $j = 0$ **to** $D-1$ **do:**")
    st.latex(r"R \sim \mathcal{U}(0,1)")
    
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**If** $\\text{Iter} < T_{\\max}/3$ **then:** *(Phase 1: Exploration)*")
    st.latex(r"s_{i,j} \gets RB_{i,j} \times (\text{Elite}_{i,j} - RB_{i,j} \times \text{Prey}_{i,j})")
    st.latex(r"\text{Prey}_{i,j} \gets \text{Prey}_{i,j} + P \times R \times s_{i,j}")
    
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Else If** $\\text{Iter} < 2T_{\\max}/3$ **then:** *(Phase 2: Transition)*")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**If** $i > N/2$ **then:**")
    st.latex(r"s_{i,j} \gets RB_{i,j} \times (RB_{i,j} \times \text{Elite}_{i,j} - \text{Prey}_{i,j})")
    st.latex(r"\text{Prey}_{i,j} \gets \text{Elite}_{i,j} + P \times CF \times s_{i,j}")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Else:**")
    st.latex(r"s_{i,j} \gets RL_{i,j} \times (\text{Elite}_{i,j} - RL_{i,j} \times \text{Prey}_{i,j})")
    st.latex(r"\text{Prey}_{i,j} \gets \text{Prey}_{i,j} + P \times R \times s_{i,j}")
    
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Else:** *(Phase 3: Exploitation)*")
    st.latex(r"s_{i,j} \gets RL_{i,j} \times (RL_{i,j} \times \text{Elite}_{i,j} - \text{Prey}_{i,j})")
    st.latex(r"\text{Prey}_{i,j} \gets \text{Elite}_{i,j} + P \times CF \times s_{i,j}")
    
    # Re-evaluation & FADs
    st.markdown("##### 4. Re-evaluation & FADs Effect")
    st.markdown("**Re-evaluate and Update Top Predator:**")
    st.markdown("**For** $i = 0$ **to** $N-1$ **do:**")
    st.latex(r"\text{Prey}[i] \gets \text{clip}(\text{Prey}[i], LB, UB)")
    st.latex(r"\text{fitness}[i] \gets f(\text{Prey}[i])")
    st.markdown("**If** $\\text{fitness}[i] < \\text{Top\\_predator\\_fit}$ **then:**")
    st.latex(r"\text{Top\_predator\_fit} \gets \text{fitness}[i], \quad \text{Top\_predator\_pos} \gets \text{Prey}[i]")
    
    st.markdown("**Marine Memory Saving (Second Time):**")
    st.markdown("*(Repeat memory saving as in Phase 1)*")
    
    st.markdown("**FADs Effect (Eddy Formation):**")
    st.markdown("**If** $\\text{random}(0,1) < FADs$ **then:**")
    st.latex(r"U \gets (\text{random}(0,1,(N,D)) < FADs) \quad \text{// Binary matrix}")
    st.latex(r"\text{Prey} \gets \text{Prey} + CF \times ((X_{\min} + \text{random}(0,1,(N,D)) \times (X_{\max} - X_{\min})) \odot U)")
    
    st.markdown("**Else:**")
    st.latex(r"r \sim \mathcal{U}(0,1)")
    st.latex(r"\text{idx}_1, \text{idx}_2 \gets \text{random\_permutations}(0, \ldots, N-1)")
    st.latex(r"s \gets (FADs \times (1-r) + r) \times (\text{Prey}[\text{idx}_1] - \text{Prey}[\text{idx}_2])")
    st.latex(r"\text{Prey} \gets \text{Prey} + s")
    
    st.markdown("**Store convergence:**")
    st.latex(r"\text{Convergence\_curve}[\text{Iter}] \gets \text{Top\_predator\_fit}")
    
    st.markdown("##### 5. Return")
    st.latex(r"\text{Return: } \text{Top\_predator\_pos}, \text{Top\_predator\_fit}, \text{Convergence\_curve}")
    
    st.markdown("---")
    
    # Download Report
    st.markdown("### 📥 Download Report")
    
    if os.path.exists("report.pdf"):
        with open("report.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="MPA_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("📄 Report PDF will be available soon. Please check back later.")

# ============================================================================
# SECTION 2: PLOTS
# ============================================================================
elif section == "Plots":
    st.markdown("## 📊 Plots & Visualizations")
    
    # Create tabs for static and dynamic plots
    tab1, tab2 = st.tabs(["📸 Static Plots", "🔄 Dynamic Convergence"])
    
    # ========================================================================
    # TAB 1: STATIC PLOTS
    # ========================================================================
    with tab1:
        st.markdown("### Static Visualization - CEC2017 Functions")
        st.markdown("""
        These plots show the landscape visualization of selected CEC2017 benchmark functions.
        Upload images for F2, F4, F12, and F25 to display them here.
        """)
        
        # Create columns for images
        
       
        st.markdown("#### F2 - F4 - F12 - F25 (Comparison VS PSO ,PSO hybride , GA )")
        if os.path.exists("plots/image.png"):
            img = Image.open("plots/image.png")
            st.image(img, use_container_width=True)
        else:
            st.info("📁 Place `image    .png` in the `plots/` folder to display")
        
            
        
    
    # ========================================================================
    # TAB 2: DYNAMIC PLOTS
    # ========================================================================
    with tab2:
        st.markdown("### Dynamic Convergence Visualization")
        st.markdown("""
        Configure the MPA parameters and select a CEC2017 benchmark function 
        to see real-time convergence visualization.
        """)
        
        if not CEC2017_AVAILABLE:
            st.error("⚠️ CEC2017 functions are not available. Please install the package from cec2017-py folder.")
            st.code("cd cec2017-py && pip install .", language="bash")
        else:
            # Create configuration columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Algorithm Parameters")
                n_agents = st.slider("Population Size (N)", 20, 100, 30, 5, 
                                    help="Number of search agents (prey)")
                dimensions = st.slider("Dimensions (D)", 10, 50, 30, 5,
                                      help="Problem dimensionality")
                max_iter = st.slider("Max Iterations", 100, 1000, 500, 50,
                                    help="Maximum number of iterations")
            
            with col2:
                st.markdown("#### MPA-Specific Parameters")
                fads_prob = st.slider("FADs Probability", 0.0, 1.0, 0.2, 0.05,
                                     help="Fish Aggregating Devices effect probability")
                p_const = st.slider("P Constant", 0.0, 1.0, 0.5, 0.05,
                                   help="Constant parameter P")
            
            with col3:
                st.markdown("#### Function Selection")
                # CEC2017 has 30 functions
                func_options = [f"F{i}" for i in range(1, 31)]
                selected_func = st.selectbox("CEC2017 Function", func_options, 
                                            index=1, help="Select benchmark function")
                
                # Extract function number
                func_num = int(selected_func[1:])
                
                # Bounds for CEC2017 are typically -100 to 100
            
            # Run button
            st.markdown("---")
            run_optimization = st.button("▶️ Run Optimization", type="primary", 
                                        use_container_width=True)
            
            if run_optimization:
                # Progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_cols = st.columns(4)
                
                # Chart placeholder
                chart_placeholder = st.empty()
                
                # Get the CEC2017 function
                cec_func = all_functions[func_num - 1]
                
                # Storage for convergence data
                iterations_list = []
                fitness_list = []
                
                # Create the initial empty chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers',
                                        name='Best Fitness',
                                        line=dict(color='#1f77b4', width=3),
                                        marker=dict(size=6)))
                fig.update_layout(
                    title=f'Convergence Curve - CEC2017 {selected_func}',
                    xaxis_title='Iteration',
                    yaxis_title='Best Fitness Value',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    xaxis=dict(range=[0, max_iter]),
                )
                
                # Initial chart display
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Callback function for real-time updates
                update_frequency = max(1, max_iter // 50)  # Update ~50 times
                
                def update_callback(iteration, best_fitness):
                    iterations_list.append(iteration)
                    fitness_list.append(best_fitness)
                    
                    # Update progress bar
                    progress = iteration / max_iter
                    progress_bar.progress(progress)
                    
                    # Update status
                    status_text.markdown(f"**Iteration:** {iteration}/{max_iter} | **Best Fitness:** {best_fitness:.6e}")
                    
                    # Update chart every N iterations
                    if iteration % update_frequency == 0 or iteration == max_iter:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=iterations_list,
                            y=fitness_list,
                            mode='lines+markers',
                            name='Best Fitness',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=6),
                            hovertemplate='Iteration: %{x}<br>Fitness: %{y:.6e}<extra></extra>'
                        ))
                        fig.update_layout(
                            title=f'Convergence Curve - CEC2017 {selected_func}',
                            xaxis_title='Iteration',
                            yaxis_title='Best Fitness Value (log scale)',
                            yaxis_type='log',
                            hovermode='x unified',
                            template='plotly_white',
                            height=500,
                            xaxis=dict(range=[0, max_iter]),
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Add delay to make visualization visible (0.1 seconds per update)
                        time.sleep(0.1)
                
                # Run the optimization
                with st.spinner("🔄 Running MPA optimization..."):
                    best_solution, best_fitness, convergence_curve = mpa_with_callback(
                        func=cec_func,
                        D=dimensions,
                        N=n_agents,
                        LB=-100,
                        UB=100,
                        Tmax=max_iter,
                        FADs=fads_prob,
                        P=p_const,
                        callback=update_callback
                    )
                
                # Final update
                progress_bar.progress(1.0)
                status_text.success(f"✅ Optimization Complete! Final Best Fitness: {best_fitness:.6e}")
                
                # Display final metrics
                with metrics_cols[0]:
                    st.metric("Final Fitness", f"{best_fitness:.6e}")
                with metrics_cols[1]:
                    st.metric("Initial Fitness", f"{convergence_curve[0]:.6e}")
                with metrics_cols[2]:
                    improvement = ((convergence_curve[0] - best_fitness) / convergence_curve[0] * 100)
                    st.metric("Improvement", f"{improvement:.2f}%")
                with metrics_cols[3]:
                    st.metric("Total Evaluations", f"{n_agents * max_iter}")
                
                # Show final results
                st.markdown("---")
                st.markdown("### 📊 Final Results")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### Convergence Statistics")
                    st.write(f"**Function:** CEC2017 {selected_func}")
                    st.write(f"**Dimensions:** {dimensions}")
                    st.write(f"**Population Size:** {n_agents}")
                    st.write(f"**Iterations:** {max_iter}")
                    st.write(f"**Initial Fitness:** {convergence_curve[0]:.6e}")
                    st.write(f"**Final Fitness:** {best_fitness:.6e}")
                    
                    # Calculate convergence speed
                    mid_fitness = convergence_curve[len(convergence_curve)//2]
                    if best_fitness < mid_fitness * 0.1:
                        speed = "Fast"
                    elif best_fitness < mid_fitness * 0.5:
                        speed = "Moderate"
                    else:
                        speed = "Slow"
                    st.write(f"**Convergence Speed:** {speed}")
                
                with col_b:
                    st.markdown("#### Best Solution (first 10 dimensions)")
                    if dimensions <= 10:
                        for i, val in enumerate(best_solution):
                            st.write(f"x[{i}] = {val:.6f}")
                    else:
                        for i in range(10):
                            st.write(f"x[{i}] = {best_solution[i]:.6f}")
                        st.write(f"... ({dimensions - 10} more dimensions)")
                
                # Download results button
                st.markdown("---")
                results_text = f"""MPA Optimization Results
Function: CEC2017 {selected_func}
Dimensions: {dimensions}
Population Size: {n_agents}
Max Iterations: {max_iter}
FADs Probability: {fads_prob}
P Constant: {p_const}

Final Best Fitness: {best_fitness:.6e}
Initial Fitness: {convergence_curve[0]:.6e}
Improvement: {improvement:.2f}%

Best Solution:
{best_solution}

Convergence Curve:
Iteration, Fitness
"""
                for i, fit in enumerate(convergence_curve):
                    results_text += f"{i+1}, {fit:.6e}\n"
                
                st.download_button(
                    label="📥 Download Results",
                    data=results_text,
                    file_name=f"mpa_results_{selected_func}_D{dimensions}.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Configure parameters above and click '▶️ Run Optimization' to start")

# ============================================================================
# SECTION 3: REFERENCES
# ============================================================================
elif section == "References":
    st.markdown("## 📚 References")
    
    st.markdown("""
    ### Primary Reference
    
    **[1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020)**  
    *Marine Predators Algorithm: A nature-inspired metaheuristic.*  
    Expert Systems with Applications, 152, 113377.  
    [https://doi.org/10.1016/j.eswa.2020.113377](https://doi.org/10.1016/j.eswa.2020.113377)
    
    
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p><b>Marine Predators Algorithm</b> - Nature-Inspired Optimization</p>
    <p style='font-size: 0.85rem;'>© 2025 | Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
