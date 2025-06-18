import streamlit as st
import pandas as pd
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import numpy as np 
from hugging_face_model import load_model_from_huggingface
from contamination_analysis import contamination

# Set page configuration
st.set_page_config(
    page_title="Contaminant Identification & Reforestation Forecasting",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c7fb8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #28a745;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        text-align: justify;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c7fb8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/14575/14575652.png", width=300)
    st.markdown("## Navigation")
    page = st.radio("Select a page:", ["Contamination Analysis", "Reforestation Forecasting"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application visualizes greenhouse gas emissions data and forecasts 
    reforestation potential to combat climate change.
    
    Data sources: Environmental monitoring datasets and reforestation models.
    """)

# Main content
st.markdown('<h1 class="main-header">🌍 Environmental Impact Dashboard</h1>', unsafe_allow_html=True)

if page == "Contamination Analysis":

    contamination()
# Replace the problematic make_subplots section with this corrected version
else:  # Reforestation Forecasting
    st.markdown('<h2 class="section-header">Reforestation Forecasting for Lima Metropolitan Area</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="description">This section provides forecasting models for reforestation efforts and their potential impact on reducing PM10 concentration in Lima Metropolitan Area. Explore different scenarios to understand how tree planting initiatives can improve air quality over time.</p>', unsafe_allow_html=True)
    
    # Load model and functions
    try:
        # Create placeholder for model loading
        with st.spinner("Loading prediction model..."):
            # Load the model directly from the path instead of file upload
                   # Hugging Face model URL
            HF_MODEL_URL = "https://huggingface.co/MilkoM0/air_quality_model/resolve/main/modelo_hibrido_new_0.9269.pkl"
        
            try:
                modelo = load_model_from_huggingface(HF_MODEL_URL)
                # Add loading of the original data
                df_original = pd.read_csv('https://raw.githubusercontent.com/Milko0/air_quality_project/b192d79a4bade1d29bac44821d6f40a37e806804/Project/prediccion_reforestacion/data/data_monitoreo_contaminantes_filtrado_best.csv')
                st.success("Model loaded successfully!")
            except Exception as e:
                st.warning(f"Could not load model from default path: {str(e)}")
                st.info("Using sample data for visualization. For accurate predictions, ensure the model file is in the correct location.")
                
            # Constants for the simulation
            TASA_SUPERVIVENCIA = 0.7
            VOLUMEN_AIRE = 2819000000 * 10  # área de Lima * altura promedio de mezcla
            
            # Historical data of PM10 in Lima
            historico_pm10 = {
                2015: 68.51, 2016: 71.85, 2017: 73.25, 2018: 69.36, 
                2019: 56.26, 2020: 53.22, 2021: 60.80, 2022: 56.51,
                2023: 41.89, 2024: 46.72
            }
            
            # SERPAR tree planting data
            arboles_serpar = {
                2020: 82637,
                2021: 160932,
                2022: 150136,
                2023: 86957,
                2024: 115548
            }
            
            ARBOLES_PROMEDIO = np.mean(list(arboles_serpar.values()))
                
            # For demo purposes - necessary functions
            def crear_datos_prediccion(start_year=2025, end_year=2035):
                """
                Crea DataFrame con todas las combinaciones necesarias para la predicción
                """
                try:
                    # If original data is available, use it
                    if 'df_original' in locals():
                        estaciones = df_original['ESTACION'].unique()
                        distritos = df_original['DISTRITO'].unique()
                    else:
                        # Fallback for demo
                        estaciones = ['Sample']
                        distritos = ['Sample']
                    
                    datos = []
                    for año in range(start_year, end_year + 1):
                        for mes in range(1, 13):
                            for estacion, distrito in zip(estaciones, distritos):
                                for hora in range(24):
                                    dia_semana = pd.Timestamp(year=año, month=mes, day=1).weekday()
                                    es_fin_semana = 1 if dia_semana >= 5 else 0
                                    datos.append({
                                        'ESTACION': estacion,
                                        'DISTRITO': distrito,
                                        'HORA': hora,
                                        'DIA_SEMANA': dia_semana,
                                        'MES': mes,
                                        'AÑO': año,
                                        'ES_FIN_DE_SEMANA': es_fin_semana
                                    })
                    return pd.DataFrame(datos)
                except Exception as e:
                    st.warning(f"Error creating prediction data: {str(e)}")
                    # Create simplified data structure for demo
                    datos = []
                    for año in range(start_year, end_year + 1):
                        for mes in range(1, 13):
                            for hora in range(24):
                                dia_semana = pd.Timestamp(year=año, month=mes, day=1).weekday()
                                es_fin_semana = 1 if dia_semana >= 5 else 0
                                datos.append({
                                    'ESTACION': 'Sample',
                                    'DISTRITO': 'Sample',
                                    'HORA': hora,
                                    'DIA_SEMANA': dia_semana,
                                    'MES': mes,
                                    'AÑO': año,
                                    'ES_FIN_DE_SEMANA': es_fin_semana
                                })
                    return pd.DataFrame(datos)
            
            def calcular_reduccion_por_arbol():
                """
                Calcula la reducción de PM10 por árbol en μg/m³
                """
                # Reducción anual por árbol en gramos
                reduccion_anual_gramos = 5217.4
                # Convertir a reducción diaria
                reduccion_diaria_gramos = reduccion_anual_gramos / 365
                # Convertir a reducción de concentración en μg/m³
                reduccion_concentracion = (reduccion_diaria_gramos * 1000) / VOLUMEN_AIRE
                return reduccion_concentracion
            
            def simular_escenarios(df_pred, arboles_anuales, start_year=2025, tasa_supervivencia=0.7):
                """
                Simula escenarios con una cantidad específica de árboles plantados anualmente
                """
                # Check if model is available for prediction
                if 'modelo' in locals() and not isinstance(modelo, type(None)):
                    try:
                        prediccion_base = modelo.predict(df_pred)
                    except Exception as e:
                        st.warning(f"Error using model for prediction: {str(e)}")
                        # Fallback to simulated data
                        prediccion_base = np.ones(len(df_pred)) * 50  # Base prediction around 50 μg/m³
                        noise = np.random.normal(0, 2, len(df_pred))  
                        years = df_pred['AÑO'].values - start_year
                        trend = years.reshape(-1, 1) * 0.5  # Increasing trend
                        prediccion_base += noise + trend.flatten()
                else:
                    # Simulated prediction for demo
                    prediccion_base = np.ones(len(df_pred)) * 50  # Base prediction around 50 μg/m³
                    noise = np.random.normal(0, 2, len(df_pred))  
                    years = df_pred['AÑO'].values - start_year
                    trend = years.reshape(-1, 1) * 0.5  # Increasing trend
                    prediccion_base += noise + trend.flatten()
                
                prediccion_con_arboles = prediccion_base.copy()
                
                # Calcular reducción por árbol
                reduccion_por_arbol = calcular_reduccion_por_arbol()
                
                años_unicos = sorted(df_pred['AÑO'].unique())
                arboles_acumulados = {}
                
                for i, año in enumerate(años_unicos):
                    # Árboles acumulados considerando supervivencia
                    n_arboles = int(arboles_anuales * tasa_supervivencia * (i + 1))
                    arboles_acumulados[año] = n_arboles
                    
                    # Reducción total por todos los árboles acumulados
                    reduccion_concentracion = reduccion_por_arbol * n_arboles
                    
                    mask_año = df_pred['AÑO'] == año
                    prediccion_con_arboles[mask_año] = np.maximum(0, 
                        prediccion_con_arboles[mask_año] - reduccion_concentracion)
                
                return prediccion_base, prediccion_con_arboles, arboles_acumulados
            
            def calcular_arboles_necesarios(df_pred, nivel_objetivo, año_objetivo=2035, max_iteraciones=20):
                """
                Calcula cuántos árboles anuales se necesitan para alcanzar un nivel objetivo de PM10
                """
                # Starting point
                arboles_min = ARBOLES_PROMEDIO  # Punto de partida: el promedio actual de SERPAR
                arboles_max = arboles_min * 10  # un límite superior inicial
                
                # Si con el máximo inicial no alcanzamos el objetivo, aumentamos el límite
                _, pred_arboles_max, _ = simular_escenarios(df_pred, arboles_max)
                mask_año_objetivo = df_pred['AÑO'] == año_objetivo
                pm10_con_max_arboles = np.mean(pred_arboles_max[mask_año_objetivo])
                
                # Si aún con el máximo no alcanzamos el objetivo, aumentamos más
                iteracion = 0
                while pm10_con_max_arboles > nivel_objetivo and iteracion < 5:
                    arboles_max *= 2
                    _, pred_arboles_max, _ = simular_escenarios(df_pred, arboles_max)
                    pm10_con_max_arboles = np.mean(pred_arboles_max[mask_año_objetivo])
                    iteracion += 1
                
                # Binary search to find the optimal number of trees
                for _ in range(max_iteraciones):
                    arboles_med = (arboles_min + arboles_max) / 2
                    _, pred_arboles_med, _ = simular_escenarios(df_pred, arboles_med)
                    mask_año_objetivo = df_pred['AÑO'] == año_objetivo
                    pm10_con_arboles_med = np.mean(pred_arboles_med[mask_año_objetivo])
                    
                    if abs(pm10_con_arboles_med - nivel_objetivo) < 0.1:
                        # Convergence found
                        return int(arboles_med)
                    
                    if pm10_con_arboles_med > nivel_objetivo:
                        # Need more trees
                        arboles_min = arboles_med
                    else:
                        # Can reduce trees
                        arboles_max = arboles_med
                
                return int((arboles_min + arboles_max) / 2)
            
            def categorizar_calidad_aire(valor):
                """
                Categoriza la calidad del aire según el valor de PM10
                """
                if valor <= 54:
                    return 'Buena'
                elif valor <= 154:
                    return 'Moderada'
                elif valor <= 254:
                    return 'Insalubre para grupos sensibles'
                else:
                    return 'Insalubre'
        
        # Main forecasting section
        st.markdown('<h3 class="section-header">PM10 Reduction Simulation</h3>', unsafe_allow_html=True)
        
        # Layout with two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Reforestation Parameters")
            
            # User input parameters
            years_forecast = st.slider("Forecast Period (years)", 5, 15, 10)
            end_year = 2025 + years_forecast
            
            # Quality target selector
            target_option = st.radio(
                "Air Quality Target",
                ["Current Trend", "Moderate Quality", "Good Quality"],
                help="Select the target air quality level you want to achieve by the end of the forecast period"
            )
            
            # Map selection to PM10 levels
            if target_option == "Current Trend":
                nivel_objetivo = 40.5
                st.info("Continuing with current reforestation rates from SERPAR")
            elif target_option == "Moderate Quality":
                nivel_objetivo = 40.5
                st.info("Targeting moderate air quality (PM10 ≤ 40.5 μg/m³)")
            else:  # Good Quality
                nivel_objetivo = 27
                st.info("Targeting good air quality (PM10 ≤ 27 μg/m³)")
                
            # Survival rate
            tasa_supervivencia = st.slider("Tree Survival Rate (%)", 50, 90, int(TASA_SUPERVIVENCIA*100)) / 100
            
            # Manual override of trees per year
            custom_trees = st.checkbox("Custom number of trees per year")
            if custom_trees:
                arboles_anuales = st.number_input("Trees to plant annually", 
                                               min_value=10000, 
                                               max_value=1000000, 
                                               value=int(ARBOLES_PROMEDIO),
                                               step=10000,
                                               format="%d")
            else:
                if target_option == "Current Trend":
                    arboles_anuales = ARBOLES_PROMEDIO
                else:
                    # Calculate required trees (in real app this would use the actual model)
                    df_prediccion = crear_datos_prediccion(2025, end_year)
                    arboles_anuales = calcular_arboles_necesarios(df_prediccion, nivel_objetivo, end_year)
            
            # Display information
            st.markdown("---")
            st.markdown("### Tree Planting History")
            st.write("SERPAR historical tree planting:")
            
            # Display historical tree planting as a table
            hist_df = pd.DataFrame({
                'Year': list(arboles_serpar.keys()),
                'Trees Planted': list(arboles_serpar.values())
            })
            st.table(hist_df)
            
            # Info about reduction per tree
            reduccion_por_arbol = calcular_reduccion_por_arbol()
            st.markdown(f"**PM10 reduction per tree:** {reduccion_por_arbol:.8f} μg/m³")
            
        with col2:
            # Create prediction data
            df_prediccion = crear_datos_prediccion(2025, end_year)
            
            # Run simulation
            pred_base, pred_arboles, arboles_acumulados = simular_escenarios(
                df_prediccion, arboles_anuales, 2025, tasa_supervivencia
            )
            
            # Calculate yearly averages
            years = list(range(2025, end_year + 1))
            pm10_sin_arboles = []
            pm10_con_arboles = []
            arboles_acum_list = []
            
            for año in years:
                mask_año = df_prediccion['AÑO'] == año
                pm10_sin_arboles.append(np.mean(pred_base[mask_año]))
                pm10_con_arboles.append(np.mean(pred_arboles[mask_año]))
                arboles_acum_list.append(arboles_acumulados[año])
            
            # Create the main forecasting visualization - fixed version without 'heights' parameter
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Historical and Projected PM10 Levels in Lima', 
                                             'Accumulated Trees for Reforestation'),
                              vertical_spacing=0.15,
                              row_heights=[0.7, 0.3])  # Using row_heights instead of heights
            
            # Add historical data
            fig.add_trace(
                go.Scatter(
                    x=list(historico_pm10.keys()),
                    y=list(historico_pm10.values()),
                    name='Historical PM10',
                    line=dict(color='gray', dash='dash'),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Add base prediction (no trees)
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=pm10_sin_arboles,
                    name='Projected PM10 (no trees)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Add prediction with trees
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=pm10_con_arboles,
                    name=f'PM10 with {int(arboles_anuales):,} trees/year',
                    line=dict(color='green', width=3)
                ),
                row=1, col=1
            )
            
            # Add margin of error (5%)
            margen_error = 0.05
            
            # Add upper/lower bounds for projection with trees
            upper_bound = [val * (1 + margen_error) for val in pm10_con_arboles]
            lower_bound = [val * (1 - margen_error) for val in pm10_con_arboles]
            
            fig.add_trace(
                go.Scatter(
                    x=years + years[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(0,128,0,0.2)',
                    line=dict(color='rgba(0,128,0,0)'),
                    name='Error margin (±5%)',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add reference lines for air quality levels
            fig.add_hline(y=54, line_dash="dash", line_color="blue",
                         annotation_text="Good quality limit", row=1, col=1)
            fig.add_hline(y=154, line_dash="dash", line_color="orange",
                         annotation_text="Moderate quality limit", row=1, col=1)
            
            # Add target level line
            fig.add_hline(y=nivel_objetivo, line_dash="dot", line_color="green",
                         annotation_text=f"Target: {nivel_objetivo} μg/m³", row=1, col=1)
            
            # Add accumulated trees bar chart
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=arboles_acum_list,
                    name='Accumulated Trees',
                    marker_color='green'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='PM10 Reduction Forecast with Reforestation in Lima Metropolitan Area',
                xaxis2_title='Year',
                yaxis_title='PM10 Concentration (μg/m³)',
                yaxis2_title='Number of Trees',
                height=700,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Impact analysis section
        st.markdown("### Reforestation Impact Analysis")
        
        # Calculate final PM10 values and reductions
        final_pm10_no_trees = pm10_sin_arboles[-1]
        final_pm10_with_trees = pm10_con_arboles[-1]
        total_reduction = final_pm10_no_trees - final_pm10_with_trees
        total_trees = arboles_acum_list[-1]
        
        # Display impact metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Final PM10 Level", 
                    f"{final_pm10_with_trees:.2f} μg/m³", 
                    f"-{total_reduction:.2f} μg/m³",
                    delta_color="inverse")
            
            # Add air quality category
            quality_cat = categorizar_calidad_aire(final_pm10_with_trees)
            st.markdown(f"<div style='text-align: center; font-weight: bold; color: {'green' if quality_cat=='Buena' else 'orange' if quality_cat=='Moderada' else 'red'};'>{quality_cat}</div>", unsafe_allow_html=True)
        
        with cols[1]:
            st.metric("Total Trees Required", 
                    f"{total_trees:,}", 
                    f"{arboles_anuales:,}/year")
        
        with cols[2]:
            # Calculate health benefit (simplified estimation)
            # Assuming reduction in respiratory issues proportional to PM10 reduction
            health_benefit = (total_reduction / final_pm10_no_trees) * 100
            st.metric("Health Benefit Estimate", 
                    f"{health_benefit:.1f}% reduction", 
                    "in respiratory issues")
            
        # Add detailed yearly projections
        with st.expander("View Detailed Yearly Projections"):
            # Create a DataFrame for the detailed projections
            results_df = pd.DataFrame({
                'Year': years,
                'PM10 without Trees': pm10_sin_arboles,
                'PM10 with Trees': pm10_con_arboles,
                'Reduction': [a-b for a, b in zip(pm10_sin_arboles, pm10_con_arboles)],
                'Accumulated Trees': arboles_acum_list,
                'Air Quality': [categorizar_calidad_aire(val) for val in pm10_con_arboles]
            })
            
            # Display as a formatted table
            st.dataframe(results_df.style.format({
                'PM10 without Trees': '{:.2f}',
                'PM10 with Trees': '{:.2f}',
                'Reduction': '{:.2f}',
                'Accumulated Trees': '{:,.0f}'
            }))
            
            # Download button for the data
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Projection Data",
                data=csv,
                file_name="reforestation_forecast.csv",
                mime="text/csv",
            )
# Reforestation strategies section
        st.markdown('<h3 class="section-header">Research-Based Reforestation Strategies for Lima</h3>', unsafe_allow_html=True)
        
        # Add evidence-based introduction
        st.markdown("""
        <div style="background-color:#e8f5e8; padding:15px; border-radius:10px; margin-bottom:20px; border-left: 4px solid #28a745;">
            <strong>Scientific Evidence:</strong> Recent studies show that species like <em>Ficus pertusa</em> (amatillo) and <em>Eucalyptus camaldulensis</em> 
            provide higher contributions to pollutant reduction in Lima, while native species like <em>Erythrina falcata</em> 
            show significant carbon storage capacity (1.27 t/individual).
        </div>
        """, unsafe_allow_html=True)
        
        strategy_tabs = st.tabs(["High-Efficiency Species", "Native Lima Species", "SERPAR Recommended"])
        
        with strategy_tabs[0]:
            st.markdown("### High-Efficiency Pollutant Reduction Species")
            st.markdown("*Based on scientific studies in Lima's urban environment*")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Primary Species (Highest PM10 Reduction):**
                
                🌳 **Eucalyptus (Eucalyptus camaldulensis)**
                - PM10 removal: 5,217.4 g/year/tree
                - High adaptation to Lima's climate
                - Fast growth rate
                - Proven effectiveness in urban environments
                
                🌳 **Amatillo (Ficus pertusa)**
                - Excellent pollutant absorption capacity
                - Dense canopy structure
                - Year-round foliage
                - Urban heat island mitigation
                """)
            
            with col2:
                st.markdown("""
                **Secondary Species (Moderate-High Efficiency):**
                
                🌳 **Red Eucalyptus (Eucalyptus camaldulensis Dehnh.)**
                - Demonstrated air purification capacity
                - Drought tolerance
                - Suitable for Lima's arid conditions
                
                🌳 **Weeping Fig (Ficus benjamina)**
                - Common in Lima urban areas
                - Good pollutant filtration
                - Ornamental value
                
                **Benefits:**
                - Scientifically proven pollutant reduction rates
                - Adapted to Lima's urban microclimate
                - Established maintenance protocols
                """)
            
        with strategy_tabs[1]:
            st.markdown("### Lima Native and Lomas Species")
            st.markdown("*Species naturally adapted to Lima's coastal desert ecosystem*")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Native Trees with Carbon Storage:**
                
                🌱 **Ceibo (Erythrina falcata)**
                - Carbon storage: 1.27 tons/individual
                - Native to Peru's coastal regions
                - Nitrogen-fixing capabilities
                - Cultural significance
                
                🌱 **Huarango (Prosopis pallida)**
                - Extremely drought resistant
                - Deep root system
                - Nitrogen fixation
                - Traditional use in coastal Peru
                """)
            
            with col2:
                st.markdown("""
                **Lomas Ecosystem Species:**
                
                🌿 **Atriplex rotundifolia**
                - Adapted to Lima's fog conditions
                - Low maintenance requirements
                - Salt tolerance
                
                🌺 **Begonia octopetala**
                - Endemic to Lima lomas
                - Year-round flowering
                - Attracts native pollinators
                
                🌸 **Ismene amancaes (Amancaes)**
                - Lima's symbolic flower
                - Fog-dependent ecosystem
                - Cultural heritage value
                """)
                
            st.info("🔬 Research shows native lomas species are specifically adapted to Lima's unique fog-dependent microclimate.")
            
        with strategy_tabs[2]:
            st.markdown("### SERPAR Implementation Strategy")
            st.markdown("*Current official reforestation program approach*")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                **Lima Verde Program Species:**
                
                The Municipal Program coordinates with 30 districts:
                - Ancón, Ate, Barranco, Breña
                - Carabayllo, Chorrillos, Comas
                - Independencia, La Molina, La Victoria
                - Los Olivos, Lurigancho-Chosica
                - Pachacámac, San Juan de Lurigancho
                - Villa El Salvador, among others
                
                **Standard Species Mix:**
                - Eucalyptus varieties (primary)
                - Ficus species (shade trees)
                - Ornamental flowering trees
                - Fruit trees in appropriate zones
                """)
            
            with col2:
                st.markdown("""
                **Implementation Approach:**
                
                ✅ **Established Protocol:**
                - Site assessment and soil analysis
                - Species selection based on location
                - Community engagement programs
                - Maintenance scheduling
                
                ✅ **Success Factors:**
                - Proven survival rates in Lima conditions
                - Municipal support and maintenance
                - Community involvement
                - Systematic monitoring
                
                **Annual Targets:**
                - Historical average: ~119,000 trees/year
                - Peak performance: 160,932 trees (2021)
                - Consistent district partnerships
                """)
            
    except Exception as e:
        st.error(f"Error in simulation: {str(e)}")
        st.info("Using the example data for visualization purposes.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Plotly | Data last updated: April 2025")