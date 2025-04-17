import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io
import base64

# Page configuration
st.set_page_config(
    page_title="DoE Manufacturing Yield Simulator",
    page_icon="🏭",
    layout="wide",
)

# Header
st.title("🏭 Manufacturing Yield Simulator")
st.markdown("""
This interactive tool simulates manufacturing process yields based on key input factors. 
Explore how different process parameters affect output and learn about Design of Experiments (DoE) concepts.
""")

# Sidebar for global settings
with st.sidebar:
    st.header("Simulator Settings")
    
    # Model complexity
    st.subheader("Model Settings")
    noise_level = st.slider("Process Noise Level (%)", 0, 10, 2, 
                           help="Controls the random variation in results - higher means more unpredictable process")
    
    # Coefficient settings (advanced)
    if st.checkbox("Show Advanced Settings", False):
        st.markdown("#### Factor Effect Coefficients")
        st.markdown("Adjust how strongly each factor affects the yield")
        
        speed_coef = st.number_input("Speed Effect", -0.5, 0.5, 0.1, 0.01)
        pressure_coef = st.number_input("Pressure Effect", -0.5, 0.5, -0.3, 0.01)
        temp_coef = st.number_input("Temperature Effect", -0.5, 0.5, 0.2, 0.01)
        material_coef = st.number_input("Material Effect", 0.0, 5.0, 2.5, 0.1)
        
        st.markdown("#### Interaction Coefficients")
        speed_pressure_coef = st.number_input("Speed × Pressure", -0.2, 0.2, 0.05, 0.01)
        speed_temp_coef = st.number_input("Speed × Temperature", -0.2, 0.2, -0.04, 0.01)
        pressure_temp_coef = st.number_input("Pressure × Temperature", -0.2, 0.2, 0.06, 0.01)
    else:
        # Default values
        speed_coef = 0.1
        pressure_coef = -0.3
        temp_coef = 0.2
        material_coef = 2.5
        speed_pressure_coef = 0.05
        speed_temp_coef = -0.04
        pressure_temp_coef = 0.06
    
    # Help information
    st.markdown("---")
    with st.expander("About DoE Concepts"):
        st.markdown("""
        **Main Effects** - The direct effect of changing one factor on the response
        
        **Interactions** - When the effect of one factor depends on the level of another factor
        
        **Response Surface** - Visualizes how factors combine to affect the response
        
        **Noise** - Random variation in the process that can't be controlled
        """)

# Main interface with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Run", "Multiple Experiments", "DoE Analysis", "Response Surface"])

# Tab 1: Single experiment run
with tab1:
    st.header("Single Experiment Run")
    st.markdown("Adjust the factors below to see their effect on manufacturing yield")

    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        speed = st.slider("Machine Speed (mm/s)", 100, 300, 200)
        pressure = st.slider("Pressure (psi)", 10, 50, 30)
        
    with col2:
        temperature = st.slider("Temperature (°C)", 20, 80, 50)
        material = st.selectbox("Material Type", ['A', 'B', 'C'])
    
    # Additional parameter
    cycle_time = st.slider("Cycle Time (seconds)", 10, 60, 30)
    
    # Encode categorical material
    material_map = {'A': 0, 'B': 1, 'C': 2}
    material_val = material_map[material]
    
    # Enhanced yield formula with more factors and interactions
    base_yield = (
        80
        + speed_coef * (speed - 200)
        + pressure_coef * (pressure - 30)
        + temp_coef * (temperature - 50)
        + material_coef * material_val
        + speed_pressure_coef * (speed - 200) * (pressure - 30)  # speed-pressure interaction
        + speed_temp_coef * (speed - 200) * (temperature - 50)   # speed-temperature interaction
        + pressure_temp_coef * (pressure - 30) * (temperature - 50)  # pressure-temperature interaction
        - 0.002 * ((cycle_time - 30) ** 2)  # nonlinear effect
        + np.random.normal(0, noise_level)  # controllable noise 
    )
    
    # Keep yield in realistic range
    base_yield = np.clip(base_yield, 50, 100)
    
    # Display result with larger font
    st.markdown(f"<h2 style='text-align: center;'>📈 Simulated Yield: {base_yield:.2f}%</h2>", unsafe_allow_html=True)
    
    # Show gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = base_yield,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Process Yield (%)"},
        gauge = {
            'axis': {'range': [50, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [50, 70], 'color': "red"},
                {'range': [70, 85], 'color': "orange"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation of current yield based on factors
    st.subheader("Process Parameter Analysis")
    st.write(f"""
    **Yield Analysis:**
    - Speed contribution: {'Positive' if speed_coef * (speed - 200) > 0 else 'Negative'} effect ({speed_coef * (speed - 200):.2f}%)
    - Pressure contribution: {'Positive' if pressure_coef * (pressure - 30) > 0 else 'Negative'} effect ({pressure_coef * (pressure - 30):.2f}%)
    - Temperature contribution: {'Positive' if temp_coef * (temperature - 50) > 0 else 'Negative'} effect ({temp_coef * (temperature - 50):.2f}%)
    - Material contribution: {material_coef * material_val:.2f}% (Material {material})
    - Main interaction effects: {speed_pressure_coef * (speed - 200) * (pressure - 30) + speed_temp_coef * (speed - 200) * (temperature - 50) + pressure_temp_coef * (pressure - 30) * (temperature - 50):.2f}%
    """)

# Tab 2: Multiple experiments
with tab2:
    st.header("Multiple Experiments")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        n_experiments = st.slider("Number of Experiments", 10, 500, 100)
    
    with col2:
        experiment_type = st.radio("Experiment Type", ["Random", "Grid", "Custom DoE"])
    
    with col3:
        st.write("Sample Strategy")
        if experiment_type == "Random":
            rand_strategy = st.radio("Random Sampling", ["Uniform", "Normal"])
        elif experiment_type == "Grid":
            grid_points = st.slider("Grid Points Per Factor", 2, 10, 3)
        else:
            doe_type = st.selectbox("DoE Type", ["Full Factorial", "Fractional Factorial"])
    
    # Run experiments button
    if st.button("🧪 Run Experiments", key="run_experiments"):
        data = []
        
        if experiment_type == "Random":
            # Random experiments
            for _ in range(n_experiments):
                if rand_strategy == "Uniform":
                    spd = np.random.uniform(100, 300)
                    prs = np.random.uniform(10, 50)
                    tmp = np.random.uniform(20, 80)
                else:  # Normal distribution around center points
                    spd = np.random.normal(200, 50)
                    spd = np.clip(spd, 100, 300)
                    prs = np.random.normal(30, 10)
                    prs = np.clip(prs, 10, 50)
                    tmp = np.random.normal(50, 15)
                    tmp = np.clip(tmp, 20, 80)
                
                mat = np.random.choice([0, 1, 2])
                ctime = np.random.uniform(10, 60)
                
                # Calculate yield with same formula as single run
                yld = (
                    80
                    + speed_coef * (spd - 200)
                    + pressure_coef * (prs - 30)
                    + temp_coef * (tmp - 50)
                    + material_coef * mat
                    + speed_pressure_coef * (spd - 200) * (prs - 30)
                    + speed_temp_coef * (spd - 200) * (tmp - 50)
                    + pressure_temp_coef * (prs - 30) * (tmp - 50)
                    - 0.002 * ((ctime - 30) ** 2)
                    + np.random.normal(0, noise_level)
                )
                yld = np.clip(yld, 50, 100)
                data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
                
        elif experiment_type == "Grid":
            # Grid-based experiments
            speeds = np.linspace(100, 300, grid_points)
            pressures = np.linspace(10, 50, grid_points)
            temps = np.linspace(20, 80, grid_points)
            materials = [0, 1, 2]  # A, B, C
            
            # Sample from grid (might be too many points for full factorial)
            sample_size = min(n_experiments, grid_points**3 * 3)
            
            # Create combinations but sample if too many
            combos = []
            for s in speeds:
                for p in pressures:
                    for t in temps:
                        for m in materials:
                            combos.append((s, p, t, m))
            
            # Sample if needed
            if len(combos) > sample_size:
                selected_combos = np.random.choice(len(combos), sample_size, replace=False)
                combos = [combos[i] for i in selected_combos]
            
            for spd, prs, tmp, mat in combos:
                ctime = np.random.uniform(10, 60)  # Still random for cycle time
                
                yld = (
                    80
                    + speed_coef * (spd - 200)
                    + pressure_coef * (prs - 30)
                    + temp_coef * (tmp - 50)
                    + material_coef * mat
                    + speed_pressure_coef * (spd - 200) * (prs - 30)
                    + speed_temp_coef * (spd - 200) * (tmp - 50)
                    + pressure_temp_coef * (prs - 30) * (tmp - 50)
                    - 0.002 * ((ctime - 30) ** 2)
                    + np.random.normal(0, noise_level)
                )
                yld = np.clip(yld, 50, 100)
                data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
                
        else:  # DoE
            # For full factorial design with 2 levels
            if doe_type == "Full Factorial":
                speeds = [120, 280]  # low, high
                pressures = [15, 45]  # low, high
                temps = [25, 75]     # low, high
                materials = [0, 2]   # A, C (low, high)
                
                for spd in speeds:
                    for prs in pressures:
                        for tmp in temps:
                            for mat in materials:
                                # Center point for cycle time
                                ctime = 30
                                
                                yld = (
                                    80
                                    + speed_coef * (spd - 200)
                                    + pressure_coef * (prs - 30)
                                    + temp_coef * (tmp - 50)
                                    + material_coef * mat
                                    + speed_pressure_coef * (spd - 200) * (prs - 30)
                                    + speed_temp_coef * (spd - 200) * (tmp - 50)
                                    + pressure_temp_coef * (prs - 30) * (tmp - 50)
                                    - 0.002 * ((ctime - 30) ** 2)
                                    + np.random.normal(0, noise_level)
                                )
                                yld = np.clip(yld, 50, 100)
                                data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
                                
                # Add center points
                for _ in range(3):  # 3 center point replicates
                    spd, prs, tmp, mat = 200, 30, 50, 1  # center values (material B)
                    ctime = 30
                    
                    yld = (
                        80
                        + speed_coef * (spd - 200)
                        + pressure_coef * (prs - 30)
                        + temp_coef * (tmp - 50)
                        + material_coef * mat
                        + speed_pressure_coef * (spd - 200) * (prs - 30)
                        + speed_temp_coef * (spd - 200) * (tmp - 50)
                        + pressure_temp_coef * (prs - 30) * (tmp - 50)
                        - 0.002 * ((ctime - 30) ** 2)
                        + np.random.normal(0, noise_level)
                    )
                    yld = np.clip(yld, 50, 100)
                    data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
                    
            else:  # Fractional factorial - half fraction of full factorial
                # Generate a half-fraction by using relation D = ABC
                speeds = [120, 280]      # Factor A
                pressures = [15, 45]     # Factor B
                temps = [25, 75]         # Factor C
                
                # Only generate half the design points
                for spd_idx, spd in enumerate(speeds):
                    for prs_idx, prs in enumerate(pressures):
                        for tmp_idx, tmp in enumerate(temps):
                            # Generator relation: mat_idx = spd_idx * prs_idx * tmp_idx (modulo 2)
                            # This ensures a half fraction
                            if (spd_idx + prs_idx + tmp_idx) % 2 == 0:
                                mat = 0  # A
                            else:
                                mat = 2  # C
                            
                            ctime = 30
                            
                            yld = (
                                80
                                + speed_coef * (spd - 200)
                                + pressure_coef * (prs - 30)
                                + temp_coef * (tmp - 50)
                                + material_coef * mat
                                + speed_pressure_coef * (spd - 200) * (prs - 30)
                                + speed_temp_coef * (spd - 200) * (tmp - 50)
                                + pressure_temp_coef * (prs - 30) * (tmp - 50)
                                - 0.002 * ((ctime - 30) ** 2)
                                + np.random.normal(0, noise_level)
                            )
                            yld = np.clip(yld, 50, 100)
                            data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
                            
                # Add center points
                for _ in range(3):
                    spd, prs, tmp, mat = 200, 30, 50, 1  # center values
                    ctime = 30
                    
                    yld = (
                        80
                        + speed_coef * (spd - 200)
                        + pressure_coef * (prs - 30)
                        + temp_coef * (tmp - 50)
                        + material_coef * mat
                        + speed_pressure_coef * (spd - 200) * (prs - 30)
                        + speed_temp_coef * (spd - 200) * (tmp - 50)
                        + pressure_temp_coef * (prs - 30) * (tmp - 50)
                        - 0.002 * ((ctime - 30) ** 2)
                        + np.random.normal(0, noise_level)
                    )
                    yld = np.clip(yld, 50, 100)
                    data.append([spd, prs, tmp, ['A', 'B', 'C'][mat], ctime, yld])
        
        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Speed", "Pressure", "Temperature", "Material", "Cycle_Time", "Yield"])
        
        # Store in session state
        st.session_state.experiment_data = df
        
        # Display results
        st.subheader("Results")
        st.dataframe(df)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Yield", f"{df['Yield'].mean():.2f}%")
            st.metric("Minimum Yield", f"{df['Yield'].min():.2f}%")
            
        with col2:
            st.metric("Maximum Yield", f"{df['Yield'].max():.2f}%")
            st.metric("Std Deviation", f"{df['Yield'].std():.2f}")
        
        # Visualization options
        st.subheader("Data Visualization")
        viz_type = st.selectbox("Visualization Type", ["3D Scatter", "Heatmap", "Box Plots"])
        
        if viz_type == "3D Scatter":
            fig = px.scatter_3d(df, x="Speed", y="Pressure", z="Yield", color="Material",
                              hover_data=["Temperature", "Cycle_Time"],
                              title="Yield Across Different Settings")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Heatmap":
            # Create pivot table
            x_axis = st.selectbox("X-Axis", ["Speed", "Pressure", "Temperature"])
            y_axis = st.selectbox("Y-Axis", ["Pressure", "Temperature", "Speed"])
            
            if x_axis != y_axis:
                # Create bin edges for continuous variables
                x_bins = pd.cut(df[x_axis], 5)
                y_bins = pd.cut(df[y_axis], 5)
                
                # Create pivot table with bins
                pivot = df.pivot_table(
                    values="Yield", 
                    index=y_bins, 
                    columns=x_bins, 
                    aggfunc="mean"
                )
                
                # Create heatmap
                fig = px.imshow(pivot, 
                               labels=dict(x=x_axis, y=y_axis, color="Yield (%)"),
                               title=f"Average Yield Heatmap: {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select different variables for X and Y axes")
                
        else:  # Box plots
            factor = st.selectbox("Factor to Analyze", ["Material", "Speed", "Pressure", "Temperature"])
            
            if factor == "Material":
                fig = px.box(df, x="Material", y="Yield", color="Material",
                           title="Yield Distribution by Material Type")
            else:
                # Create bins for continuous variables
                bins = pd.cut(df[factor], 5)
                fig = px.box(df, x=bins, y="Yield",
                           title=f"Yield Distribution by {factor} Ranges")
                
            st.plotly_chart(fig, use_container_width=True)
            
        # Download button for experiment data
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="experiment_results.csv">Download Experiment Data (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

# Tab 3: DoE Analysis
with tab3:
    st.header("DoE Analysis")
    
    if 'experiment_data' not in st.session_state:
        st.info("Please run experiments in the 'Multiple Experiments' tab first to enable analysis")
    else:
        df = st.session_state.experiment_data
        
        # Main effects plots
        st.subheader("Main Effects Plots")
        st.write("These plots show how each factor independently affects the yield.")
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=("Speed Effect", "Pressure Effect", 
                                          "Temperature Effect", "Material Effect"))
        
        # Speed effect
        speed_groups = pd.cut(df['Speed'], 5)
        speed_effects = df.groupby(speed_groups)['Yield'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=[str(interval) for interval in speed_effects['Speed']], 
                      y=speed_effects['Yield'], mode='lines+markers'),
            row=1, col=1
        )
        
        # Pressure effect
        pressure_groups = pd.cut(df['Pressure'], 5)
        pressure_effects = df.groupby(pressure_groups)['Yield'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=[str(interval) for interval in pressure_effects['Pressure']], 
                      y=pressure_effects['Yield'], mode='lines+markers'),
            row=1, col=2
        )
        
        # Temperature effect
        temp_groups = pd.cut(df['Temperature'], 5)
        temp_effects = df.groupby(temp_groups)['Yield'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=[str(interval) for interval in temp_effects['Temperature']], 
                      y=temp_effects['Yield'], mode='lines+markers'),
            row=2, col=1
        )
        
        # Material effect
        material_effects = df.groupby('Material')['Yield'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=material_effects['Material'], y=material_effects['Yield']),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interaction plots
        st.subheader("Interaction Plots")
        st.write("These plots show how factors interact with each other to affect the yield.")
        
        # Choose factors to analyze interactions
        col1, col2 = st.columns(2)
        with col1:
            factor1 = st.selectbox("First Factor", ["Speed", "Pressure", "Temperature", "Material"])
        with col2:
            factor2 = st.selectbox("Second Factor", ["Pressure", "Temperature", "Material", "Speed"])
            
        if factor1 != factor2:
            # Create interaction plot
            if factor1 == "Material" or factor2 == "Material":
                # Handle categorical variable differently
                if factor1 == "Material":
                    cat_var, num_var = factor1, factor2
                else:
                    cat_var, num_var = factor2, factor1
                    
                # Create bins for numerical variable
                df['bin'] = pd.cut(df[num_var], 3, labels=['Low', 'Medium', 'High'])
                
                fig = px.line(df.groupby(['bin', cat_var])['Yield'].mean().reset_index(), 
                            x=cat_var, y='Yield', color='bin',
                            title=f"Interaction Plot: {factor1} × {factor2}",
                            markers=True)
                    
            else:
                # Both factors are numerical
                df['bin1'] = pd.cut(df[factor1], 3, labels=['Low', 'Medium', 'High'])
                df['bin2'] = pd.cut(df[factor2], 3, labels=['Low', 'Medium', 'High'])
                
                fig = px.line(df.groupby(['bin1', 'bin2'])['Yield'].mean().reset_index(), 
                            x='bin1', y='Yield', color='bin2',
                            title=f"Interaction Plot: {factor1} × {factor2}",
                            labels={'bin1': factor1, 'bin2': factor2},
                            markers=True)
                
            st.plotly_chart(fig, use_container_width=True)
            
            if st.checkbox("Show Statistical Analysis"):
                st.subheader("Statistical Model")
                
                # Create formula for model
                if factor1 == "Material" or factor2 == "Material":
                    formula = f"Yield ~ C({factor1}) * {factor2}" if factor1 == "Material" else f"Yield ~ {factor1} * C({factor2})"
                else:
                    formula = f"Yield ~ {factor1} * {factor2}"
                
                # Fit model
                model = ols(formula, data=df).fit()
                st.write("Model Summary:")
                st.write(model.summary().tables[1])
                
                # Check if interaction term is significant
                p_values = model.pvalues
                interaction_term = [term for term in p_values.index if ':' in term][0] if ':' in str(p_values.index) else None
                
                if interaction_term and p_values[interaction_term] < 0.05:
                    st.success(f"✅ Significant interaction detected between {factor1} and {factor2} (p={p_values[interaction_term]:.4f})")
                else:
                    st.info("❓ No significant interaction detected")
        else:
            st.error("Please select two different factors for interaction analysis")
            
        # Pareto chart of effects
        st.subheader("Pareto Chart of Effects")
        
        # Fit a model with all main effects and interactions
        model_formula = "Yield ~ Speed + Pressure + Temperature + C(Material) + Speed:Pressure + Speed:Temperature + Pressure:Temperature"
        full_model = ols(model_formula, data=df).fit()
        
        # Extract coefficients and p-values
        coeffs = full_model.params[1:]  # Skip intercept
        p_values = full_model.pvalues[1:]
        
        # Create DataFrame for plotting
        effect_data = pd.DataFrame({
            'Effect': coeffs.index,
            'Absolute_Coefficient': abs(coeffs.values),
            'P_Value': p_values.values,
            'Significant': p_values.values < 0.05
        })
        
        # Sort by absolute coefficient
        effect_data = effect_data.sort_values('Absolute_Coefficient', ascending=False)
        
        # Create Pareto chart
        fig = px.bar(effect_data, x='Effect', y='Absolute_Coefficient', 
                    color='Significant',
                    color_discrete_map={True: 'darkblue', False: 'lightgray'},
                    title="Pareto Chart of Standardized Effects")
        
        # Add significance line
        fig.add_shape(type="line", x0=-0.5, x1=len(effect_data)-0.5, y0=2.0, y1=2.0,
                    line=dict(color="red", width=2, dash="dash"))
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Response Surface
with tab4:
    st.header("Response Surface Methodology")
    st.write("Explore the response surface to find optimal settings for your process.")
    
    if 'experiment_data' not in st.session_state:
        st.info("Please run experiments in the 'Multiple Experiments' tab first")
    else:
        df = st.session_state.experiment_data
        
        # Select factors for response surface
        col1, col2 = st.columns(2)
        
        with col1:
            x_factor = st.selectbox("X-Axis Factor", ["Speed", "Pressure", "Temperature"])
        
        with col2:
            y_factor = st.selectbox("Y-Axis Factor", ["Pressure", "Temperature", "Speed"])
            
        if x_factor != y_factor:
            # Create filter for other factors if needed
            material_filter = st.selectbox("Material Filter", ["All Materials", "Material A", "Material B", "Material C"])
            
            filtered_df = df.copy()
            if material_filter != "All Materials":
                material_type = material_filter.split(" ")[-1]
                filtered_df = df[df["Material"] == material_type]
                
            # Create a grid for the response surface
            x_range = np.linspace(filtered_df[x_factor].min(), filtered_df[x_factor].max(), 50)
            y_range = np.linspace(filtered_df[y_factor].min(), filtered_df[y_factor].max(), 50)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            
            # Fit a model using polynomial regression
            formula = f"Yield ~ {x_factor} + {y_factor} + np.power({x_factor}, 2) + np.power({y_factor}, 2) + {x_factor}:{y_factor}"
            model = ols(formula, data=filtered_df).fit()
            
            # Predict on the grid
            grid_df = pd.DataFrame({x_factor: x_grid.flatten(), y_factor: y_grid.flatten()})
            z_pred = model.predict(grid_df)
            z_grid = z_pred.values.reshape(x_grid.shape)
            
            # Create contour plot
            fig = go.Figure(data=[
                go.Surface(x=x_range, y=y_range, z=z_grid, colorscale='Viridis')
            ])
            
            # Add actual data points
            fig.add_trace(go.Scatter3d(
                x=filtered_df[x_factor],
                y=filtered_df[y_factor],
                z=filtered_df['Yield'],
                mode='markers',
                marker=dict(size=4, color='red'),
                name='Actual Data'
            ))
            
            fig.update_layout(
                title=f"Response Surface: {x_factor} vs {y_factor}",
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title='Yield (%)'
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Contour plot
            st.subheader("2D Contour Plot")
            
            fig = go.Figure(data=
                go.Contour(
                    z=z_grid,
                    x=x_range,
                    y=y_range,
                    colorscale='Viridis',
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    )
                )
            )
            
            # Add actual data points
            fig.add_trace(go.Scatter(
                x=filtered_df[x_factor],
                y=filtered_df[y_factor],
                mode='markers',
                marker=dict(color='white', size=8, line=dict(color='black', width=1)),
                name='Experiments'
            ))
            
            fig.update_layout(
                title=f"Contour Plot: Yield vs {x_factor} and {y_factor}",
                xaxis_title=x_factor,
                yaxis_title=y_factor,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimal point
            if st.checkbox("Find Optimal Settings"):
                # Find maximum point in the grid
                max_idx = np.argmax(z_grid)
                optimal_x = x_grid.flatten()[max_idx]
                optimal_y = y_grid.flatten()[max_idx]
                optimal_yield = z_grid.flatten()[max_idx]
                
                st.success(f"**Optimal Settings Found:**\n\n"
                          f"- {x_factor}: {optimal_x:.2f}\n"
                          f"- {y_factor}: {optimal_y:.2f}\n"
                          f"- Predicted Yield: {optimal_yield:.2f}%")
                
                # Show model details
                if st.checkbox("Show Model Details"):
                    st.write("Response Surface Model:")
                    st.write(model.summary().tables[1])
        else:
            st.error("Please select two different factors for the response surface")

# Footer with information
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
<p><b>DoE Manufacturing Yield Simulator</b><br>
Created for IA4.0 Smart Factory Project<br>
Version 1.0 - April 2025</p>
</div>
""", unsafe_allow_html=True)

