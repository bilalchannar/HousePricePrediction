import os
import subprocess
import sys


def show_menu():
    print("\nHouse Price Prediction ML Project\n")
    print("1. Run scraper")
    print("2. Run preprocessing")
    print("3. Train models")
    print("4. Predict house price (Console)")
    print("5. Predict house price (GUI)")
    print("6. Run Streamlit Web Application")
    print("7. Show model charts")
    print("8. Evaluate best model")
    print("9. Exit")


def run_file(file_path):
    try:
        subprocess.run([sys.executable, file_path], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to run: {file_path}")
    except KeyboardInterrupt:
        print(f"\nExecution of {os.path.basename(file_path)} was interrupted by the user.")


def run_streamlit():
    try:
        # Run streamlit in a subprocess
        subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception:
        print("Failed to run Streamlit. Make sure it is installed and in your PATH.")


def main():
    while True:
        show_menu()
        try:
            choice = input("Enter your choice: ").strip()
        except KeyboardInterrupt:
            print("\nExiting program.")
            break

        if choice == "1":
            run_file(os.path.join("src", "scraper.py"))
        elif choice == "2":
            run_file(os.path.join("src", "preprocess.py"))
        elif choice == "3":
            run_file(os.path.join("src", "train_models.py"))
        elif choice == "4":
            run_file(os.path.join("src", "predict_console.py"))
        elif choice == "5":
            run_file(os.path.join("src", "predict_gui.py"))
        elif choice == "6":
            run_streamlit()
        elif choice == "7":
            run_file(os.path.join("src", "visualize_results.py"))
        elif choice == "8":
            run_file(os.path.join("src", "evaluate_best_model.py"))
        elif choice == "9":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
        sys.exit(0)

