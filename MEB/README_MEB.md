# 🧮 MEB Calculation Tool - Streamlit Application

A full-featured Streamlit web application for calculating Minimum Expenditure Basket (MEB) based on TUIK (Turkish Statistical Institute) inflation indices.

## 📋 Features

### 🏠 Three Main Pages:

1. **📖 Introduction**
   - Overview of MEB concept
   - Links to official reports
   - Explanation of MEB components

2. **📤 Data Upload & Calculation**
   - Upload TUIK Excel files
   - Select from 4 scenarios: ESSN, CESSN, Ineligible, Turkish
   - Calculate MEB with different quantity baskets
   - View results in multiple detailed tables
   - Download results as Excel file

3. **📊 Graphs & Visualization**
   - Interactive Plotly charts
   - Filter by components and date range
   - Bar charts showing component breakdown
   - Line charts for trend analysis
   - Inflation rate visualization
   - Summary statistics (average, max, min, growth %)

## 🚀 Installation

### Requirements:
- Python 3.8+
- pip

### Setup:

```bash
# 1. Install dependencies
pip install -r requirements_meb.txt

# 2. Run the application
streamlit run meb_calculator.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Scenarios

The application supports 4 different MEB calculation scenarios:

| Scenario | Description | Food Quantity | Utilities |
|----------|-------------|---------------|-----------|
| **ESSN** | Essential Scenario | Higher | Standard |
| **CESSN** | Core ESSN | Medium | Standard |
| **Ineligible** | Non-eligible basket | Higher | Standard |
| **Turkish** | Turkey-specific | Same as Ineligible | Higher utilities |

## 📁 Data Format

### Required Excel File Structure:
- Data should be TUIK inflation index data
- File should have product codes as columns (01111, 01113, etc.)
- First 4 rows are skipped (headers)
- Last 5 rows are removed

### Supported Product Codes:
- **Food**: 01111 (Rice/Bulgur), 01113 (Bread), 01146 (Yogurt), etc.
- **Health**: 06111 (Medicine), 06231 (Doctor fees)
- **Housing**: 04110 (Rent), 04411 (Water), 04510 (Electricity), 04522 (Gas)
- **Education**: 09740 (Notebooks/Stationery)
- **Personal Care**: 13120 (Various hygiene products)
- **Transportation**: 07321 (Bus), 08320 (Phone)

## 🎨 Features Breakdown

### Data Processing:
✅ Reads TUIK Excel files  
✅ Calculates inflation rates (year-over-year)  
✅ Applies quantity multipliers by scenario  
✅ Generates summary statistics  

### Output Tables:
- **MEB Summary**: Total by category (Food, Rent, Utilities, etc.)
- **Components Detail**: All individual items
- **Food Items**: Breakdown of food products
- **Shelter**: Housing and utilities costs
- **Other Items**: Education, health, transportation

### Visualization:
- 📊 Component comparison (bar chart)
- 📈 Total MEB trend (line chart)
- 💹 Inflation rates (multi-line chart)
- 📌 Summary statistics (metrics)

## 💾 Export Options

- Download results as **Excel** with multiple sheets:
  - MEB_Calculated: Summary totals
  - MEB_by_Items: Detailed component breakdown
  - Price_Data: Raw price data
  - Multipliers: Calculation multipliers

## 🔧 Configuration

### MEB Quantities by Scenario:
Located in `MEB_QUANTITIES` dictionary in the code. Customize quantities for each product and scenario as needed.

### Color Scheme:
Colors can be customized in the visualization section:
```python
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", ...]
```

## 📝 Usage Example

1. **Open the app**: `streamlit run meb_calculator.py`
2. **Go to Introduction**: Review MEB concept and components
3. **Go to Data Upload**:
   - Click "Upload TUIK Indexes Excel File"
   - Select scenario (e.g., "ESSN")
   - Click "Calculate MEB"
   - View results in tabs
   - Download Excel file
4. **Go to Graphs**:
   - Select components to compare
   - Choose date range
   - Click "Generate Charts"
   - Analyze visualizations

## 🐛 Troubleshooting

### File Upload Issues:
- Ensure file is .xlsx or .xls format
- Check that file has proper TUIK structure
- File should not be corrupted

### Calculation Errors:
- Verify product codes exist in your data
- Check for empty columns
- Ensure numeric data is properly formatted

### Performance:
- Limit date range for faster chart rendering
- Use fewer components for cleaner visualization

## 📚 References

- [Turkish Statistical Institute (TUIK)](https://www.tuik.gov.tr/)
- [Kızılay Platform](https://platform.kizilaykart.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

## 📄 License

This application is provided as-is for MEB calculation purposes.

## 👨‍💻 Support

For issues or feature requests, please contact the development team.

---

**Version**: 1.0  
**Last Updated**: February 6, 2026  
**Built with**: Streamlit + Plotly + Pandas
