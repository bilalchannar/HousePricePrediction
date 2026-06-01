import os
import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from utils import format_price_pkr, engineer_features_dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
LOC_AVG_PATH = os.path.join(BASE_DIR, "models", "location_avg_price.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_zameen_lahore.csv")

FEATURE_COLUMNS = [
    "area_marla",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "built_in_year",
    "parking_space",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
    "property_age",
    "total_rooms",
    "luxury_score",
    "bedroom_bathroom_ratio",
    "location_avg_price"
]

# Global states
entries = {}
error_labels = {}
unit_var = None
result_var = None
root_window = None
result_label = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", "Model file not found. Please train the model first.")
        return None
    return joblib.load(MODEL_PATH)

def load_location_averages():
    if not os.path.exists(LOC_AVG_PATH):
        return {}
    return joblib.load(LOC_AVG_PATH)

def load_best_model_rmse(default_val=45000000.0):
    try:
        results_path = os.path.join(BASE_DIR, "models", "model_results.csv")
        if os.path.exists(results_path):
            rdf = pd.read_csv(results_path)
            # Find row with lowest RMSE or highest R2
            best_row = rdf.sort_values(by="r2_score", ascending=False).iloc[0]
            return float(best_row["rmse"])
    except Exception:
        pass
    return default_val

def get_combobox_values(column_name, fallback_values):
    """Loads unique column values from processed CSV if available."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            if column_name in df.columns:
                vals = sorted(df[column_name].dropna().unique().tolist())
                if "Other" not in vals and column_name == "location":
                    vals.append("Other")
                return vals
    except Exception:
        pass
    return fallback_values

def clear_errors():
    for label in error_labels.values():
        label.config(text="")

def validate_fields():
    clear_errors()
    has_errors = False
    
    # 1. Area Validation
    area_str = entries["area"].get().strip()
    if not area_str:
        error_labels["area"].config(text="Required")
        has_errors = True
    else:
        try:
            val = float(area_str)
            if val <= 0:
                error_labels["area"].config(text="Must be > 0")
                has_errors = True
        except ValueError:
            error_labels["area"].config(text="Must be a number")
            has_errors = True
            
    # 2. Bedrooms Validation
    beds_str = entries["bedrooms"].get().strip()
    if not beds_str:
        error_labels["bedrooms"].config(text="Required")
        has_errors = True
    else:
        try:
            val = float(beds_str)
            if val < 0 or val > 20:
                error_labels["bedrooms"].config(text="Must be 0-20")
                has_errors = True
        except ValueError:
            error_labels["bedrooms"].config(text="Must be a number")
            has_errors = True
            
    # 3. Bathrooms Validation
    baths_str = entries["bathrooms"].get().strip()
    if not baths_str:
        error_labels["bathrooms"].config(text="Required")
        has_errors = True
    else:
        try:
            val = float(baths_str)
            if val < 0 or val > 20:
                error_labels["bathrooms"].config(text="Must be 0-20")
                has_errors = True
        except ValueError:
            error_labels["bathrooms"].config(text="Must be a number")
            has_errors = True
            
    # 4. Built-in Year Validation
    year_str = entries["built_in_year"].get().strip()
    if year_str:
        try:
            val = int(float(year_str))
            if val != 0 and (val < 1900 or val > 2026):
                error_labels["built_in_year"].config(text="Must be 1900-2026")
                has_errors = True
        except ValueError:
            error_labels["built_in_year"].config(text="Must be a year")
            has_errors = True

    # 5. Numeric validation for rest of the numeric entries
    other_fields = ["parking_space", "servant_quarters", "store_rooms", "kitchens", "drawing_rooms"]
    for field in other_fields:
        val_str = entries[field].get().strip()
        if val_str:
            try:
                float(val_str)
            except ValueError:
                error_labels[field].config(text="Must be a number")
                has_errors = True
                
    return not has_errors

def predict_price():
    if not validate_fields():
        return
        
    model = load_model()
    if model is None:
        return
        
    loc_averages = load_location_averages()
    rmse = load_best_model_rmse()

    # Get inputs and convert area according to selected unit
    area_val = float(entries["area"].get().strip())
    unit = unit_var.get()
    
    if unit == "Kanal":
        area_marla = area_val * 20.0
    elif unit == "Square Feet":
        area_marla = area_val / 272.25
    elif unit == "Square Yards":
        area_marla = area_val / 30.25
    else: # Marla
        area_marla = area_val

    # Parse numerical details
    bedrooms = float(entries["bedrooms"].get().strip())
    bathrooms = float(entries["bathrooms"].get().strip())
    built_in_year = float(entries["built_in_year"].get().strip() or 2020)
    parking_space = float(entries["parking_space"].get().strip() or 0)
    servant_quarters = float(entries["servant_quarters"].get().strip() or 0)
    store_rooms = float(entries["store_rooms"].get().strip() or 0)
    kitchens = float(entries["kitchens"].get().strip() or 1)
    drawing_rooms = float(entries["drawing_rooms"].get().strip() or 1)
    
    location = entries["location"].get().strip() or "Unknown"
    property_type = entries["property_type"].get().strip() or "House"

    # Assemble raw dict
    raw_input = {
        "area_marla": area_marla,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "property_type": property_type,
        "built_in_year": built_in_year,
        "parking_space": parking_space,
        "servant_quarters": servant_quarters,
        "store_rooms": store_rooms,
        "kitchens": kitchens,
        "drawing_rooms": drawing_rooms
    }

    # Engineer features
    engineered_dict = engineer_features_dict(raw_input, loc_averages)
    input_data = pd.DataFrame([engineered_dict], columns=FEATURE_COLUMNS)

    try:
        result_var.set("Predicting price...")
        root_window.update_idletasks()
        
        # Predict price (outputs log scaled price)
        prediction_log = model.predict(input_data)[0]
        predicted_price = int(round(float(np.expm1(prediction_log))))
        
        # Calculate Range and Confidence
        lower_price = max(500000, predicted_price - rmse)
        upper_price = predicted_price + rmse
        
        # Determine confidence level based on location familiarity
        if location in loc_averages and location != "Other" and location != "Unknown":
            confidence = "High"
        else:
            confidence = "Medium"
            
        # Price per Marla
        price_per_marla = predicted_price / area_marla if area_marla > 0 else 0
        
        # Build formatting response card text
        result_text = (
            f"Estimated House Price: {format_price_pkr(predicted_price)}\n"
            f"Expected Range: {format_price_pkr(lower_price)} - {format_price_pkr(upper_price)}\n"
            f"Confidence: {confidence}  |  Price per Marla: Rs. {int(price_per_marla):,}"
        )
        result_var.set(result_text)
        
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed. Error details: {e}")

def load_sample():
    # Set form fields with sample values
    entries["area"].delete(0, tk.END)
    entries["area"].insert(0, "10")
    unit_var.set("Marla")
    
    entries["bedrooms"].delete(0, tk.END)
    entries["bedrooms"].insert(0, "4")
    
    entries["bathrooms"].delete(0, tk.END)
    entries["bathrooms"].insert(0, "5")
    
    entries["location"].set("DHA Defence")
    entries["property_type"].set("House")
    
    entries["built_in_year"].delete(0, tk.END)
    entries["built_in_year"].insert(0, "2022")
    
    entries["parking_space"].delete(0, tk.END)
    entries["parking_space"].insert(0, "2")
    
    entries["servant_quarters"].delete(0, tk.END)
    entries["servant_quarters"].insert(0, "1")
    
    entries["store_rooms"].delete(0, tk.END)
    entries["store_rooms"].insert(0, "1")
    
    entries["kitchens"].delete(0, tk.END)
    entries["kitchens"].insert(0, "2")
    
    entries["drawing_rooms"].delete(0, tk.END)
    entries["drawing_rooms"].insert(0, "1")
    
    clear_errors()
    result_var.set("Sample loaded. Click 'Predict Price' to check estimation.")

def reset_form():
    for key, entry in entries.items():
        if isinstance(entry, ttk.Combobox):
            entry.set("Other" if key == "location" else "House")
        else:
            entry.delete(0, tk.END)
            if key in ["kitchens", "drawing_rooms"]:
                entry.insert(0, "1")
            elif key == "built_in_year":
                entry.insert(0, "2020")
            else:
                entry.insert(0, "0")
    unit_var.set("Marla")
    clear_errors()
    result_var.set("Estimated House Price will appear here")

def add_field(parent, label_text, row, key, default_text="", is_combobox=False, combo_values=None):
    label = tk.Label(parent, text=label_text, bg="#f4f6f8", fg="#222222", font=("Arial", 11))
    label.grid(row=row, column=0, sticky="w", padx=10, pady=8)

    if is_combobox:
        entry = ttk.Combobox(parent, values=combo_values or [], font=("Arial", 11), state="normal")
        entry.set(default_text)
    else:
        entry = tk.Entry(parent, font=("Arial", 11), bd=1, relief="solid")
        entry.insert(0, default_text)
        
    entry.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
    
    # Validation error label
    err_label = tk.Label(parent, text="", bg="#f4f6f8", fg="#d32f2f", font=("Arial", 9, "bold"))
    err_label.grid(row=row, column=2, sticky="w", padx=5, pady=8)
    
    entries[key] = entry
    error_labels[key] = err_label
    return entry

def create_gui():
    global result_var, result_label, root_window, unit_var

    root_window = tk.Tk()
    root_window.title("House Price Prediction System")
    root_window.geometry("780x700")
    root_window.configure(bg="#f4f6f8")
    root_window.resizable(True, True)

    # Style configuration for ttk comboboxes
    style = ttk.Style()
    style.theme_use('clam')

    header = tk.Frame(root_window, bg="#1f4e79", height=90)
    header.pack(fill="x")
    header.pack_propagate(False)

    title = tk.Label(
        header,
        text="House Price Prediction System",
        bg="#1f4e79",
        fg="white",
        font=("Arial", 20, "bold"),
    )
    title.pack(pady=(15, 2))

    subtitle = tk.Label(
        header,
        text="Enter property details to estimate the house price",
        bg="#1f4e79",
        fg="white",
        font=("Arial", 10),
    )
    subtitle.pack()

    body_frame = tk.Frame(root_window, bg="#f4f6f8")
    body_frame.pack(fill="both", expand=True, padx=20, pady=(20, 8))

    canvas = tk.Canvas(body_frame, bg="#f4f6f8", highlightthickness=0)
    scrollbar = tk.Scrollbar(body_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f4f6f8")

    scrollable_frame.bind(
        "<Configure>",
        lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Bind mouse wheel scroll
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

    form_frame = scrollable_frame
    form_frame.columnconfigure(1, weight=1)
    form_frame.columnconfigure(2, minsize=140)

    # Load locations and property types list dynamically
    locations_list = get_combobox_values("location", ["DHA Defence", "Bahria Town", "Johar Town", "Lake City", "GT Road Area", "Sabzazar Scheme", "Other"])
    prop_types_list = get_combobox_values("property_type", ["House", "Flat", "Penthouse", "Room", "Upper Portion", "Lower Portion"])

    # 1. Custom Area Selector Row
    label = tk.Label(form_frame, text="Area", bg="#f4f6f8", fg="#222222", font=("Arial", 11))
    label.grid(row=0, column=0, sticky="w", padx=10, pady=8)
    
    area_inner_frame = tk.Frame(form_frame, bg="#f4f6f8")
    area_inner_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=8)
    area_inner_frame.columnconfigure(0, weight=3)
    area_inner_frame.columnconfigure(1, weight=1)
    
    area_entry = tk.Entry(area_inner_frame, font=("Arial", 11), bd=1, relief="solid")
    area_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
    area_entry.insert(0, "0")
    entries["area"] = area_entry
    
    unit_var = tk.StringVar(value="Marla")
    unit_dropdown = ttk.Combobox(area_inner_frame, textvariable=unit_var, values=["Marla", "Kanal", "Square Feet", "Square Yards"], font=("Arial", 10), state="readonly")
    unit_dropdown.grid(row=0, column=1, sticky="ew")
    
    err_label = tk.Label(form_frame, text="", bg="#f4f6f8", fg="#d32f2f", font=("Arial", 9, "bold"))
    err_label.grid(row=0, column=2, sticky="w", padx=5, pady=8)
    error_labels["area"] = err_label

    # Other Rows
    add_field(form_frame, "Bedrooms", 1, "bedrooms", "0")
    add_field(form_frame, "Bathrooms", 2, "bathrooms", "0")
    add_field(form_frame, "Location", 3, "location", "Other", is_combobox=True, combo_values=locations_list)
    add_field(form_frame, "Property Type", 4, "property_type", "House", is_combobox=True, combo_values=prop_types_list)
    add_field(form_frame, "Built in Year", 5, "built_in_year", "2020")
    add_field(form_frame, "Parking Space", 6, "parking_space", "0")
    add_field(form_frame, "Servant Quarters", 7, "servant_quarters", "0")
    add_field(form_frame, "Store Rooms", 8, "store_rooms", "0")
    add_field(form_frame, "Kitchens", 9, "kitchens", "1")
    add_field(form_frame, "Drawing Rooms", 10, "drawing_rooms", "1")

    # Bottom Area
    footer_frame = tk.Frame(root_window, bg="#f4f6f8")
    footer_frame.pack(fill="x", padx=20, pady=(5, 15))

    # Control Button panel
    button_panel = tk.Frame(footer_frame, bg="#f4f6f8")
    button_panel.pack(fill="x", pady=(0, 10))
    button_panel.columnconfigure((0, 1, 2, 3), weight=1)

    predict_btn = tk.Button(
        button_panel,
        text="Predict Price",
        command=predict_price,
        bg="#2e7d32",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#256528",
        activeforeground="white",
        bd=0,
        pady=8
    )
    predict_btn.grid(row=0, column=0, sticky="ew", padx=4)

    sample_btn = tk.Button(
        button_panel,
        text="Load Sample",
        command=load_sample,
        bg="#1976d2",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#1565c0",
        activeforeground="white",
        bd=0,
        pady=8
    )
    sample_btn.grid(row=0, column=1, sticky="ew", padx=4)

    reset_btn = tk.Button(
        button_panel,
        text="Reset Form",
        command=reset_form,
        bg="#f57c00",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#e65100",
        activeforeground="white",
        bd=0,
        pady=8
    )
    reset_btn.grid(row=0, column=2, sticky="ew", padx=4)

    exit_btn = tk.Button(
        button_panel,
        text="Exit App",
        command=root_window.quit,
        bg="#d32f2f",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#c62828",
        activeforeground="white",
        bd=0,
        pady=8
    )
    exit_btn.grid(row=0, column=3, sticky="ew", padx=4)

    # Pricing details card
    result_card = tk.Frame(footer_frame, bg="#e8f5e9", bd=1, relief="solid")
    result_card.pack(fill="x")

    result_label_title = tk.Label(
        result_card,
        text="Estimation Results Card",
        bg="#e8f5e9",
        fg="#222222",
        font=("Arial", 11, "bold"),
    )
    result_label_title.pack(anchor="w", padx=12, pady=(10, 2))

    result_var = tk.StringVar(value="Estimated House Price will appear here")
    result_label = tk.Label(
        result_card,
        textvariable=result_var,
        bg="#e8f5e9",
        fg="#1b5e20",
        font=("Arial", 12, "bold"),
        wraplength=700,
        justify="left",
    )
    result_label.pack(anchor="w", padx=12, pady=(0, 12))

    root_window.mainloop()


if __name__ == "__main__":
    create_gui()
