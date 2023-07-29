import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import base64

hide_menu = """
<style>
#MainMenu {
    visibility:visible;
}
footer{
    visibility:visible;
}
footer:after{
    content:'Copyright @2023: Rakesh Suthar. for more info, email:rakeshsuthar1996@gmail.com';
    display:block;
    position:relative;
    color:blue;
    padding:1px;
}
<style>
"""




# Custom header content
st.set_page_config(page_title='EQE and voltage loss analysis Calculator', page_icon='📈')
st.title("EQE-Jsc and Voltage-loss analysis calculator")
st.markdown(
    """
    <div class="social-links">
        <a href="www.linkedin.com/in/rakesh-suthar-125b3b136" target="_blank">LinkedIn</a>
        <a href="https://twitter.com/Rakeshsuthar645" target="_blank">Twitter</a>
        <a href="https://www.researchgate.net/profile/Rakesh-Suthar" target="_blank">ResearchGate</a>
        <a href="https://ivparameters.streamlit.app/" target="_blank">Photovoltaic parameters calculator</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(hide_menu, unsafe_allow_html=True)
st.caption("App developed by [Rakesh Suthar, IIT Delhi](https://sites.google.com/view/rakeshiitd/home)")

# Load AM1.5 data
am15_data = pd.read_csv('AM15G.csv')
am15_wavelength = am15_data['Wavelength']
am15_intensity = am15_data['Intensity']
am15_energy = 1240 / am15_wavelength
# Convert AM1.5 intensity to flux (AM1.5G / energy eV)
am15_flux = am15_intensity / (1240 / am15_wavelength)

# Load the CSV data for the phiBB
data = pd.read_csv('PhiiBB.csv')
phiBB_wavelength = data['Wavelength'].values
phiBB_intensity = data['Intensity'].values

# Function to format the value to three digits
def format_value(val):
    if isinstance(val, (int, float)):
        return f"{val:.3f}"
    else:
        return val

# Function to interpolate data and ensure step size of 1 for 'Wavelength' and 'EQE'
def interpolate_and_update_data(original_data):
    x_original = original_data['Wavelength'].values
    y_original = original_data['EQE'].values

    step_size = np.diff(x_original).min()

    if step_size != 1:
        x_new = np.arange(x_original.min(), x_original.max() + 1)
        y_new = np.interp(x_new, x_original, y_original)
        original_data = pd.DataFrame({'Wavelength': x_new, 'EQE': y_new})

    return original_data

# Calculate Jsc_SQ based on the SQ limit and given bandgap (in eV)?????????????????????????????????????????????????????????????????????????????????????????????
def calculate_jsc_sq(bandgap_eV, eqe_percent):
    # Convert EQE percentage to decimal fraction
    eqe = eqe_percent / 100.0
    jsc_sq_values = []
    jsc_sq = 0.0
    for energy, flux in zip(am15_wavelength, am15_flux):
        # Calculate the photon energy (in eV)
        photon_energy_eV = 1240 / energy
        # Determine the EQE value based on the photon energy and bandgap
        if photon_energy_eV >= bandgap_eV:
            eqe = eqe  # Use the given EQE percentage above the bandgap
        else:
            eqe = 0.0  # 0% EQE below the bandgap
        # Calculate the SQ limit for the current wavelength and add it to Jsc_SQ
        sq_limit = 0.1 * (flux * eqe)
        jsc_sq += sq_limit
        jsc_sq_values.append(jsc_sq)
    return jsc_sq_values


# Main Streamlit app function
def main():
    # Just for the formatting
    st.markdown(
        """
        <style>
        .header {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
        }
        .logo img {
            width: 100px;
            border-radius: 50%;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .social-links {
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }
        .social-links a {
            margin: 0 10px;
            color: #3366cc;
            font-size: 18px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input form for user to provide EQE, bandgap, and VOC values
    bandgap_eV = st.sidebar.number_input('Bandgap Energy (eV)', min_value=0.00, max_value=10.00, step=0.01, value=1.50)
    voc_voltage = st.sidebar.number_input('Voc Voltage (V)', min_value=0.00, max_value=10.00, step=0.001, value=0.800)
    st.sidebar.write("Option 1: Paste the two data columns (Wavelength(nm), EQE(%)) separated by commas or tabs:")
    direct_input = st.sidebar.text_area("Direct Input", value="", height=100)
    uploaded_file = st.sidebar.file_uploader("Option 2: Upload your .csv file, Provide two data columns (Wavelength(nm), EQE(%)) in .csv file", type=["csv"])

    eqe_data = None  # Initialize the data variable
    if uploaded_file is not None:
        # Read the uploaded data
        eqe_data = pd.read_csv(uploaded_file)
        # Automatically assign column names
        eqe_data.columns = ['Wavelength', 'EQE']  # You can change the column names if needed
        st.write("Data loaded successfully.")
    elif direct_input:
        # Convert direct input to a DataFrame
        eqe_data = pd.read_csv(StringIO(direct_input), sep='\t|,', engine='python', header=None)
        eqe_data.columns = ['Wavelength', 'EQE']
        # Convert 'Wavelength' and 'EQE' columns to numeric types
        eqe_data['Wavelength'] = pd.to_numeric(eqe_data['Wavelength'], errors='coerce')
        eqe_data['EQE'] = pd.to_numeric(eqe_data['EQE'], errors='coerce')
        # Drop rows with NaN values
        eqe_data.dropna(inplace=True)

    if eqe_data is not None:  # Calculate results and plot IV curve if data is loaded
        # Calculate IV curve and PCE
        updated_eqe_data = interpolate_and_update_data(eqe_data.copy())
        # Ensure that 'Wavelength' column in updated_eqe_data matches the AM1.5G wavelength range
        updated_eqe_data = updated_eqe_data[updated_eqe_data['Wavelength'].isin(am15_wavelength)]

        # Calculate product of EQE and AM1.5G and then integrate over wavelength using Simpson's rule
        product_eqe_am15 = 0.001 * np.interp(am15_wavelength, updated_eqe_data['Wavelength'], updated_eqe_data['EQE']) * am15_flux
        cumulative_sum = product_eqe_am15.cumsum()
        # Calculate the integrated Jsc by taking the last value in the cumulative sum
        integrated_jsc = cumulative_sum.values[-1]

        new_data = pd.DataFrame({
            'Wavelength': updated_eqe_data['Wavelength'],  # Updated Wavelength from updated_eqe_data
            'EQE': updated_eqe_data['EQE'],  # Updated EQE from updated_eqe_data
            'Cumulative Sum': cumulative_sum  # Cumulative Sum array
        })

        # Calculate Jsc_SQ for different levels of EQE (100%, 90%, 80%, ...)
        eqe_levels = [100, 90, 80, 70, 60]
        jsc_sq_values = {}
        for eqe_percent in eqe_levels:
            jsc_sq = calculate_jsc_sq(bandgap_eV, eqe_percent)
            jsc_sq_values[f'EQE {eqe_percent}%'] = jsc_sq

        # Print Jsc_SQ at given bandgap with 100% EQE
        bandgap_jsc_sq = jsc_sq_values['EQE 100%']

        # Convert wavelength to energy (eV)
        energy = 1240 / phiBB_wavelength

        # Find the index where photon energy is greater than the bandgap energy
        bandgap_index = np.argmax(energy < bandgap_eV)

        accumulated_intensity = np.trapz(phiBB_intensity[:bandgap_index], am15_wavelength[:bandgap_index])
        q = 1.6e-19  # Charge of an electron in coulombs
        j0 = accumulated_intensity * q  # Convert to mA/cm²
        voc_sq = 0.026 *np.log((bandgap_jsc_sq[-1]/ j0) + 1)


        product_eqe_phiBB = np.interp(am15_wavelength, updated_eqe_data['Wavelength'], updated_eqe_data['EQE']) * phiBB_intensity
        integrated_j0_energy = np.trapz(phiBB_intensity[:bandgap_index], am15_wavelength[:bandgap_index])
        integrated_j0_energy_1 = 1000 * q * integrated_j0_energy

        # Calculate Voc_radiative using the equation
        voc_radiative =  0.026 * np.log((integrated_jsc) / integrated_j0_energy_1 + 1)



        #Ok all things are calculateed , now plots the results 
        # Calculate the differences (delta values)
        delta_V1 = bandgap_eV - voc_sq
        delta_V2 = voc_sq - voc_radiative
        delta_V3 = voc_radiative - voc_voltage
        delta_V = bandgap_eV-voc_voltage

        # Show the plot
        fig1 = go.Figure()

        # Add the first trace for interpolated EQE values on y1-axis
        fig1.add_trace(go.Scatter(x=updated_eqe_data['Wavelength'], y=updated_eqe_data['EQE'],
                                 mode='lines', name='Interpolated EQE', line=dict(color='blue'),
                                 yaxis='y1'))

        # Set y-axis title for interpolated EQE values
        fig1.update_layout(yaxis=dict(title='Interpolated EQE', titlefont=dict(color='blue'), tickfont=dict(color='blue')))

        # Add the second trace for cumulative sum on y2-axis
        fig1.add_trace(go.Scatter(x=am15_wavelength, y=cumulative_sum, mode='lines',
                                 name='Cumulative Sum', line=dict(color='red'),
                                 yaxis='y2'))

        # Set y-axis title for cumulative sum
        fig1.update_layout(yaxis2=dict(title='Intgrated Jsc (mA/cm²)', overlaying='y', side='right',
                                      titlefont=dict(color='red'), tickfont=dict(color='red')))

        # Add text annotation for the integrated Jsc value on the graph
        fig1.add_annotation(text=f"Integrated Jsc: {format_value(integrated_jsc)} mA/cm²",
                           xref='paper', yref='paper', x=0.5, y=0.35,
                           showarrow=False, align='right')

        fig1.update_layout(
            xaxis_title='Wavelength (nm)',
            title='EQE and Integrated Jsc Curve',
            xaxis_range=[300, 1000],
            xaxis=dict(title_font=dict(size=16, family='Arial, sans-serif', color='black'), tickfont = dict( color='black', size=17)),
            yaxis1=dict(title_font=dict(size=16, family='Arial, sans-serif', color='blue'), tickfont = dict(size=17)),
            yaxis2=dict(title_font=dict(size=16, family='Arial, sans-serif', color='red'), tickfont = dict(size=17)),
            plot_bgcolor='white',
            hovermode='closest',
            font=dict(family="Arial, sans-serif", size=15, color="black"),
            autosize=True,
            width=700,
            height=500,
            xaxis_showgrid=True,  # Show vertical gridlines
            yaxis_showgrid=False,  # Show horizontal gridlines
            margin=dict(l=50, r=50, b=50, t=50, pad=4, autoexpand=True),
            showlegend=False,
        )



        # Create a bar diagram for the values and differences
        labels = ['Eg/q', 'Voc_SQ', 'Voc_rad', 'Voc']
        values = [bandgap_eV, voc_sq, voc_radiative, voc_voltage]
        differences = [delta_V1, delta_V2, delta_V3]
        units = ['V', 'V', 'V', 'V']

        # Create a list to hold bar traces
        bar_traces = []
       # Plot the values as blue bars and display the values in the middle of the bars with rotated text
        for i, (label, value, unit) in enumerate(zip(labels, values, units)):
            bar_trace = go.Bar(
                x=[label],
                y=[value],
                name=f'{label}: {value:.3f} {unit}',
                marker=dict(color='blue'),
                text=[f'{value:.3f} {unit}'],
                textposition='inside',
                textangle=-0,
            )
            bar_traces.append(bar_trace)

        # Plot the differences as orange bars and display the differences in the middle of the bars with rotated text
        for i, (label, difference, unit) in enumerate(zip(labels[1:], differences, units[1:])):
            bar_trace = go.Bar(
                x=[label],
                y=[difference],
                name=f'Δ{label}: {difference:.3f} {unit}',
                marker=dict(color='orange'),
                text=[f'{difference:.3f} {unit}'],
                textposition='inside',  
                textangle=-00,
            )
            bar_traces.append(bar_trace)


        # Create the layout for fig2
        fig2_layout = go.Layout(
            title='Voltage Loss Analysis of Solar Cell',
            xaxis=dict(tickfont=dict(size=15, family='Arial, sans-serif', color='black')),
            yaxis=dict(title='Voltage (V)', title_font=dict(size=16, family='Arial, sans-serif', color='black'), tickfont=dict(size=16, family='Arial, sans-serif', color='black')),
            barmode='stack',
            showlegend=False,
        )

        fig2 = go.Figure(data=bar_traces, layout=fig2_layout)




        # Create a DataFrame for the results

        results_data = pd.DataFrame({
            'Parameter': ['Integrated Jsc mA/cm²', 'Jsc_SQ at bandgap value', 'Voc_SQ limit', 'Voc_rad, radiative limit', 'Voc **User input**', 'ΔV1 (Eg/q - Voc_SQ)', 'ΔV2 (Voc_SQ - Voc_rad), radiative loss', 'ΔV3 (Voc_rad - Voc) non-radiative loss', 'Total Loss (ΔV = Eg/q - Voc)'],
            'Value': [integrated_jsc, bandgap_jsc_sq[-1],  voc_sq, voc_radiative, voc_voltage, delta_V1, delta_V2, delta_V3, delta_V], 
            'Units': ['mA/cm²', 'mA/cm²', 'V','V','V','V','V','V','V']
        })

        # Format the 'Value' column to three significant digits
        results_data['Value'] = results_data['Value'].apply(format_value)

        # Display the results in a table with units, making the values in the 'Value' column bold
        st.write("Results:")
        results_data_styled = results_data.style.apply(lambda x: ["font-weight: bold" if col == 'Value' else "" for col in x], axis=1)
        st.table(results_data_styled)

       # Show the plots side by side
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)



        # Concatenate the results and the new data into a single DataFrame
        combined_data =pd.concat([results_data, new_data], ignore_index=True)

        # Download button to save all data (results + new data) as a single CSV
        csv_combined = combined_data.to_csv(index=False)
        b64_combined = base64.b64encode(csv_combined.encode()).decode()  # Convert DataFrame to base64
        href_combined = f'<a href="data:file/csv;base64,{b64_combined}" download="EQE_calculation.csv">Download All Data (Loss parameters, Wavelength, EQE and Integrated Jsc)</a>'
        st.markdown(href_combined, unsafe_allow_html=True)


        
# Run the Streamlit app
if __name__ == '__main__':
    main()
