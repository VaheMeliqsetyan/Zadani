import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
import yaml
import requests
import json

class RCS:
    def __init__(self, diameter, fmin, fmax):
        self.diameter = diameter
        self.fmin = fmin
        self.fmax = fmax
        self.radius = diameter / 2

    def calculate_rcs(self, frequency):
        wavelength = 3e8 / frequency
        k = 2 * np.pi / wavelength
        rcs_value = (wavelength**2 / np.pi) * np.abs(
            sum(
                (-1)**n * (n + 0.5) * (self.bn(n, k) - self.an(n, k))
                for n in range(1, 21)
            )
        )**2
        return rcs_value

    def an(self, n, k):
        jn = spherical_jn(n, k * self.radius)
        hn = self.hn(n, k * self.radius)
        return jn / hn

    def bn(self, n, k):
        jn_prev = spherical_jn(n - 1, k * self.radius)
        jn = spherical_jn(n, k * self.radius)
        hn_prev = self.hn(n - 1, k * self.radius)
        hn = self.hn(n, k * self.radius)
        return (k * self.radius * jn_prev - n * jn) / (k * self.radius * hn_prev - n * hn)

    def hn(self, n, x):
        jn = spherical_jn(n, x)
        yn = spherical_yn(n, x)
        return jn + 1j * yn

    def plot_rcs(self):
        frequencies = np.linspace(self.fmin, self.fmax, 500)
        rcs_values = [self.calculate_rcs(f) for f in frequencies]

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, rcs_values, label="RCS", color="red")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("RCS (m²)")
        plt.title("Radar Cross Section vs Frequency")
        plt.grid(True)
        plt.legend()
        plt.show()

        rcs_data = [{"frequency": f, "rcs": r} for f, r in zip(frequencies, rcs_values)]
        with open("rcs_results.json", "w") as json_file:
            json.dump(rcs_data, json_file, indent=2)

def load_variant_data(url, variant_number):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверяем на ошибки HTTP
        
        data = yaml.safe_load(response.text)
        if not isinstance(data, list):
            raise ValueError("YAML data should be a list of variants")
            
        for variant in data:
            if isinstance(variant, dict) and variant.get('number') == variant_number:
                try:
                    diameter = float(variant['D'])
                    fmin = float(variant['fmin'])
                    fmax = float(variant['fmax'])
                    return diameter, fmin, fmax
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Invalid data format in variant {variant_number}: {e}")
        
        raise ValueError(f"Variant {variant_number} not found in YAML.")
    
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the YAML data: {e}")
    except yaml.YAMLError as e:
        raise Exception(f"Failed to parse YAML data: {e}")

def main():
    url = "https://jenyay.net/uploads/Student/Modelling/task_rcs_02.yaml"
    variant_number = 1  # Укажите здесь ваш номер варианта
    
    try:
        diameter, fmin, fmax = load_variant_data(url, variant_number)
        print(f"Loaded data for variant {variant_number}:")
        print(f"Diameter: {diameter} m")
        print(f"Frequency range: {fmin} - {fmax} Hz")
        
        rcs = RCS(diameter, fmin, fmax)
        rcs.plot_rcs()
        
    except Exception as e:
        print(f"Error: {e}")
        # В случае ошибки можно использовать значения по умолчанию для демонстрации
        print("Using default values for demonstration...")
        rcs = RCS(0.1, 1e9, 10e9)
        rcs.plot_rcs()

if __name__ == "__main__":
    main()