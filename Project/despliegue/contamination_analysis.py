import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  

def contamination():

    st.markdown('<h2 class="section-header">Greenhouse Gas Emissions Analysis</h2>', unsafe_allow_html=True)
        
    st.markdown('<p class="description">This section presents interactive visualizations of greenhouse gas emissions by sector, category, and emission source over time. Explore the data to identify major contributors to climate change.</p>', unsafe_allow_html=True)
        
        # File uploader
    uploaded_file = st.file_uploader("Upload your emissions data (Excel format)", type="xlsx")
        
    if uploaded_file is not None:
            # Load and preprocess data
            try:
                df = pd.read_excel(uploaded_file)
                
                # Data preprocessing
                columnas_numericas = [
                    'DIOXIDO_DE_CARBONO_GGCO2',
                    'METANO_GGCH4',
                    'METANO_EQUIVALENTE_GGCO2EQ',
                    'OXIDO_NITROSO_GGN2O',
                    'OXIDO_NITROSO_EQUIVALENTE_GGCO2EQ',
                    'EMISIONES_GEI_GGCO2EQ'
                ]
                df[columnas_numericas] = df[columnas_numericas].fillna(0)
                
                # Fill NA values in categorical columns
                df['SUBCATEGORIA'] = df['SUBCATEGORIA'].fillna('Sin especificar')
                df['CATEGORIA'] = df['CATEGORIA'].fillna('Sin especificar')
                df['FUENTE_DE_EMISION'] = df['FUENTE_DE_EMISION'].fillna('Sin especificar')
                
                # Filter for valid emissions
                df = df[df["EMISIONES_GEI_GGCO2EQ"] > 0]
                
                # Get unique years
                years = sorted(df["ANIO"].unique())
                
                if not years:
                    st.error("No valid data available for visualization.")
                else:
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Hierarchical View", "Timeline Analysis", "Sector Comparison"])
                    
                    # Tab 1: Hierarchical visualizations
                    with viz_tabs[0]:
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Year selector
                            selected_year = st.selectbox("Select Year", years, key='treemap_year')
                            # Visualization type selector
                            viz_type = st.radio("Visualization Type", ["Treemap", "Sunburst"])
                        
                        with col2:
                            # Filter data for selected year
                            df_year = df[df["ANIO"] == selected_year]
                            
                            if viz_type == "Treemap":
                                # Create treemap visualization
                                fig_treemap = px.treemap(
                                    df_year,
                                    path=['SECTOR', 'CATEGORIA', 'SUBCATEGORIA'],
                                    values='EMISIONES_GEI_GGCO2EQ',
                                    color='EMISIONES_GEI_GGCO2EQ',
                                    color_continuous_scale='RdYlBu_r',
                                    title=f'Emissions by Sector and Category ({selected_year})'
                                )
                                fig_treemap.update_layout(height=700)
                                st.plotly_chart(fig_treemap, use_container_width=True)
                            
                            else:  # Sunburst
                                # Create sunburst visualization
                                fig_sunburst = px.sunburst(
                                    df_year,
                                    path=['SECTOR', 'CATEGORIA', 'FUENTE_DE_EMISION'],
                                    values='EMISIONES_GEI_GGCO2EQ',
                                    color='EMISIONES_GEI_GGCO2EQ',
                                    color_continuous_scale='Viridis',
                                    title=f'Hierarchical Analysis of GHG Emissions ({selected_year})',
                                    maxdepth=3
                                )
                                fig_sunburst.update_layout(height=700)
                                fig_sunburst.update_traces(
                                    hovertemplate="""
                                    <b>%{label}</b><br>
                                    Emissions: %{value:,.2f} GgCO₂eq<br>
                                    Percentage: %{percentRoot:.2f}%
                                    <extra></extra>
                                    """,
                                    textinfo='label+percent entry'
                                )
                                st.plotly_chart(fig_sunburst, use_container_width=True)
                    
                    # Tab 2: Timeline Analysis
                    with viz_tabs[1]:
                        # Prepare aggregated data by year
                        temporal = df.groupby('ANIO').agg({
                            'DIOXIDO_DE_CARBONO_GGCO2': 'sum',
                            'METANO_EQUIVALENTE_GGCO2EQ': 'sum',
                            'OXIDO_NITROSO_EQUIVALENTE_GGCO2EQ': 'sum'
                        }).reset_index()
                        
                        # Create temporal graph
                        fig_temporal = go.Figure()
                        
                        # Add lines for each emission type
                        fig_temporal.add_trace(go.Scatter(
                            x=temporal['ANIO'],
                            y=temporal['DIOXIDO_DE_CARBONO_GGCO2'],
                            name='CO₂',
                            mode='lines+markers',
                            line=dict(color='#ff7f0e', width=3)
                        ))
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=temporal['ANIO'],
                            y=temporal['METANO_EQUIVALENTE_GGCO2EQ'],
                            name='CH₄ (CO₂eq)',
                            mode='lines+markers',
                            line=dict(color='#1f77b4', width=3)
                        ))
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=temporal['ANIO'],
                            y=temporal['OXIDO_NITROSO_EQUIVALENTE_GGCO2EQ'],
                            name='N₂O (CO₂eq)',
                            mode='lines+markers',
                            line=dict(color='#2ca02c', width=3)
                        ))
                        
                        # Calculate total emissions
                        temporal['TOTAL_EMISSIONS'] = (
                            temporal['DIOXIDO_DE_CARBONO_GGCO2'] + 
                            temporal['METANO_EQUIVALENTE_GGCO2EQ'] + 
                            temporal['OXIDO_NITROSO_EQUIVALENTE_GGCO2EQ']
                        )
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=temporal['ANIO'],
                            y=temporal['TOTAL_EMISSIONS'],
                            name='Total GHG Emissions',
                            mode='lines+markers',
                            line=dict(color='#d62728', width=4, dash='dot')
                        ))
                        
                        # Update layout
                        fig_temporal.update_layout(
                            title='Temporal Evolution of Emissions by Gas Type',
                            xaxis_title='Year',
                            yaxis_title='Emissions (GgCO₂eq)',
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_temporal, use_container_width=True)
                        
                        # Add some insights about the temporal trends
                        with st.expander("📊 Emission Trends Analysis"):
                            # Calculate growth rates
                            first_year = temporal['ANIO'].min()
                            last_year = temporal['ANIO'].max()
                            first_total = temporal.loc[temporal['ANIO'] == first_year, 'TOTAL_EMISSIONS'].values[0]
                            last_total = temporal.loc[temporal['ANIO'] == last_year, 'TOTAL_EMISSIONS'].values[0]
                            
                            total_change_pct = ((last_total - first_total) / first_total) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Change", f"{total_change_pct:.1f}%", 
                                        delta=f"{total_change_pct:.1f}%", 
                                        delta_color="inverse")
                            
                            with col2:
                                # Find the year with highest emissions
                                max_year = temporal.loc[temporal['TOTAL_EMISSIONS'].idxmax(), 'ANIO']
                                max_emissions = temporal['TOTAL_EMISSIONS'].max()
                                st.metric("Peak Emissions Year", f"{max_year}", 
                                        f"{max_emissions:.2f} GgCO₂eq")
                            
                            with col3:
                                # Calculate average annual change
                                years_diff = last_year - first_year
                                if years_diff > 0:
                                    avg_annual_change = total_change_pct / years_diff
                                    st.metric("Avg. Annual Change", f"{avg_annual_change:.2f}%/year",
                                            delta=f"{avg_annual_change:.2f}%")
                    
                    # Tab 3: Sector Comparison
                    with viz_tabs[2]:
                        # Get sectors and prepare data
                        sectors = df['SECTOR'].unique()
                        year_range = st.select_slider(
                            "Select Year Range", 
                            options=years,
                            value=(min(years), max(years))
                        )
                        
                        # Filter data for selected year range
                        mask = (df['ANIO'] >= year_range[0]) & (df['ANIO'] <= year_range[1])
                        df_range = df[mask]
                        
                        # Aggregate emissions by sector
                        sector_emissions = df_range.groupby('SECTOR').agg({
                            'EMISIONES_GEI_GGCO2EQ': 'sum'
                        }).reset_index().sort_values('EMISIONES_GEI_GGCO2EQ', ascending=False)
                        
                        # Create bar chart
                        fig_sectors = px.bar(
                            sector_emissions,
                            x='SECTOR',
                            y='EMISIONES_GEI_GGCO2EQ',
                            color='EMISIONES_GEI_GGCO2EQ',
                            color_continuous_scale='Viridis',
                            title=f'Total Emissions by Sector ({year_range[0]} - {year_range[1]})',
                            labels={'EMISIONES_GEI_GGCO2EQ': 'GHG Emissions (GgCO₂eq)', 'SECTOR': 'Sector'}
                        )
                        
                        fig_sectors.update_layout(
                            xaxis_tickangle=-45,
                            height=600
                        )
                        
                        st.plotly_chart(fig_sectors, use_container_width=True)
                        
                        # Add sector comparison
                        cols = st.columns(2)
                        with cols[0]:
                            # Pie chart for sector distribution
                            fig_pie = px.pie(
                                sector_emissions,
                                values='EMISIONES_GEI_GGCO2EQ',
                                names='SECTOR',
                                title='Proportion of Emissions by Sector',
                                hole=0.4
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with cols[1]:
                            # Select specific sectors to compare
                            top_sectors = sector_emissions.head(5)['SECTOR'].tolist()
                            selected_sectors = st.multiselect(
                                "Select sectors to compare over time", 
                                options=sectors,
                                default=top_sectors[:3]
                            )
                            
                            if selected_sectors:
                                # Create temporal comparison for selected sectors
                                sector_time_data = df[df['SECTOR'].isin(selected_sectors)].groupby(['ANIO', 'SECTOR']).agg({
                                    'EMISIONES_GEI_GGCO2EQ': 'sum'
                                }).reset_index()
                                
                                fig_sector_time = px.line(
                                    sector_time_data,
                                    x='ANIO',
                                    y='EMISIONES_GEI_GGCO2EQ',
                                    color='SECTOR',
                                    markers=True,
                                    title='Temporal Evolution by Sector',
                                    labels={'EMISIONES_GEI_GGCO2EQ': 'Emissions (GgCO₂eq)', 'ANIO': 'Year'}
                                )
                                
                                fig_sector_time.update_layout(height=400)
                                st.plotly_chart(fig_sector_time, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing the data: {e}")
        
    else:
            # Display sample data and instructions
            st.info("Please upload your greenhouse gas emissions data file in Excel format to begin analysis.")
            st.markdown("""
            The data should contain the following columns:
            - ANIO (Year)
            - SECTOR
            - CATEGORIA
            - SUBCATEGORIA
            - FUENTE_DE_EMISION
            - DIOXIDO_DE_CARBONO_GGCO2
            - METANO_GGCH4
            - METANO_EQUIVALENTE_GGCO2EQ
            - OXIDO_NITROSO_GGN2O
            - OXIDO_NITROSO_EQUIVALENTE_GGCO2EQ
            - EMISIONES_GEI_GGCO2EQ
            """)
            
            # Sample visualization
            st.markdown("### Sample Visualization")
            st.image("https://www.epa.gov/sites/default/files/2016-05/us-ghg-2014.png", caption="Sample GHG emissions visualization")