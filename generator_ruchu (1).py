import requests
import xml.etree.ElementTree as ET
import datetime
import csv
from xml.dom import minidom

# Konfiguracja TomTom API
API_KEY = "z3LD7Zi2Pkg623mSeP8okCItGx8PzTNA"
BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Współrzędne dróg
coordinates = {
    "-E7": (51.136166, 17.069618),
    "-E4": (51.136096, 17.070020),
    "E4": (51.136100, 17.069792),
    "E7": (51.136139, 17.069691),
    "-E8": (51.135880, 17.069760),
    "E8": (51.135940, 17.069587)
}

this_time = datetime.datetime.now()

# Godziny
hours = [this_time]

CSV_FILE = "traffic_data.csv"
SIMULATION_TIME = 3600
FLOW_END_TIME = 3500

# Robienie CSV
def initialize_csv():
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["timestamp", "hour", "edge", "latitude", "longitude",
                             "currentSpeed", "freeFlowSpeed", "density"])

# Zapis danych do CSV
def save_to_csv(hour, edge, lat, lon, current_speed, free_flow_speed, density):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hour,
            edge,
            lat,
            lon,
            current_speed,
            free_flow_speed,
            density
        ])

# Pobranie natężenia ruchu z TomTom API
def get_traffic_density(lat, lon, edge, hour):
    url = f"{BASE_URL}?point={lat},{lon}&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    speed = data['flowSegmentData']['currentSpeed']
    free_flow_speed = data['flowSegmentData']['freeFlowSpeed']
    density = round(free_flow_speed / max(speed, 1), 2)

    save_to_csv(hour, edge, lat, lon, speed, free_flow_speed, density)
    return density

# Funkcja do formatowania XML
def prettify_xml(element):
    raw_str = ET.tostring(element, 'utf-8')
    parsed_str = minidom.parseString(raw_str)
    return parsed_str.toprettyxml(indent="  ")

# Generowanie pliku .rou
def generate_route_file(hour, net_file_patch="nowe_skrzyżowanie.net.xml"):
    formatted_hour = hour.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"traffic_scenario_{formatted_hour}.rou.xml"

    root = ET.Element("routes")

    for flow_id, (edge_from, edge_to) in enumerate([("-E7", "E8"), ("-E7", "-E4"), ("E4", "E7"), ("E4", "E8"), ("-E8", "-E4")]):
        lat, lon = coordinates[edge_from]
        density = get_traffic_density(lat, lon, edge_from, hour)
        period = max(1 / density, 0.1)

        ET.SubElement(root, "flow", attrib={
            "id": f"f_{flow_id}",
            "begin": "0.00",
            "from": edge_from,
            "to": edge_to,
            "end": str(FLOW_END_TIME),
            "period": str(period)
        })

    with open(filename, "w", encoding="UTF-8") as file:
        file.write(prettify_xml(root))

    print(f"Plik {filename} został wygenerowany.")
    generate_sumocfg_file(formatted_hour, filename, net_file_patch)

# Generowanie pliku .sumocfg
def generate_sumocfg_file(hour, route_file, net_file):
    filename = f"simulation_{hour}.sumocfg"
    root = ET.Element("configuration")

    input_section = ET.SubElement(root, "input")
    ET.SubElement(input_section, "net-file", value=net_file)
    ET.SubElement(input_section, "route-files", value=route_file)

    time_section = ET.SubElement(root, "time")
    ET.SubElement(time_section, "begin", value="0")
    ET.SubElement(time_section, "end", value=str(SIMULATION_TIME))

    with open(filename, "w", encoding="UTF-8") as file:
        file.write(prettify_xml(root))

    print(f"Plik {filename} został wygenerowany.")

# Inicjalizacja CSV
initialize_csv()

# Uruchomienie skryptu dla każdej godziny
for hour in hours:
    generate_route_file(hour)
