import folium
import json
import os
from typing import List, Dict, Tuple


def read_metadata_from_jsonl(file_path: str) -> List[Dict]:
    """Read metadata from JSONL file."""
    metadata = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                metadata.append(json.loads(line.strip()))
    return metadata


def calculate_center_coordinates(metadata: List[Dict]) -> Tuple[float, float]:
    """Calculate the center coordinates for the map."""
    if not metadata:
        return (52.5097, 13.3769)  # Default to Berlin coordinates

    total_lat = sum(item['lat'] for item in metadata)
    total_lon = sum(item['lon'] for item in metadata)
    center_lat = total_lat / len(metadata)
    center_lon = total_lon / len(metadata)

    return (center_lat, center_lon)


def extract_connections(metadata: List[Dict]) -> List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]]:
    """Extract unique connections between points from metadata links."""
    connections = []
    seen_connections = set()  # To avoid duplicates

    # Create a lookup dict for quick coordinate access
    coord_lookup = {item['id']: (item['lat'], item['lon']) for item in metadata}

    for item in metadata:
        current_id = item['id']
        current_coords = coord_lookup[current_id]

        # Process links
        for link in item.get('links', []):
            linked_id = link['pano']['id']
            linked_coords = coord_lookup.get(linked_id)

            if linked_coords:
                # Create a sorted tuple to avoid duplicate connections
                connection_key = tuple(sorted([current_id, linked_id]))

                if connection_key not in seen_connections:
                    connections.append((current_id, linked_id, current_coords, linked_coords))
                    seen_connections.add(connection_key)

    return connections


def create_folium_map(metadata: List[Dict], center_coords: Tuple[float, float], connections: List[Tuple]) -> folium.Map:
    """Create a Folium map with markers and connection lines."""
    # Create map centered on the calculated center
    m = folium.Map(location=center_coords, zoom_start=15)

    # Create a feature group for connections
    connection_group = folium.FeatureGroup(name="Connections", show=True)
    marker_group = folium.FeatureGroup(name="Points", show=True)

    # Add connection lines first (so they appear behind markers)
    for start_id, end_id, start_coords, end_coords in connections:
        folium.PolyLine(
            locations=[start_coords, end_coords],
            color='red',
            weight=2,
            opacity=0.7,
            tooltip=f"Connection: {start_id} ↔ {end_id}"
        ).add_to(connection_group)

    # Add markers for each point
    for i, item in enumerate(metadata):
        # Create popup content with metadata
        links_info = ""
        if item.get('links'):
            links_info = f"<b>Links:</b> {len(item['links'])} connections<br>"

        popup_content = f"""
        <b>ID:</b> {item['id']}<br>
        <b>Coordinates:</b> {item['lat']:.6f}, {item['lon']:.6f}<br>
        <b>Elevation:</b> {item.get('elevation', 'N/A')} m<br>
        <b>Date:</b> {item.get('date', 'N/A')}<br>
        <b>Heading:</b> {item.get('heading', 'N/A'):.2f}<br>
        {links_info}
        """

        # Add marker with popup
        folium.Marker(
            location=[item['lat'], item['lon']],
            popup=popup_content,
            tooltip=f"Point {i+1}: {item['id']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_group)

    # Add feature groups to map
    connection_group.add_to(m)
    marker_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def visualize_points_on_map(jsonl_file_path: str, output_html_path: str = None):
    """Main function to visualize points from JSONL file on Folium map."""
    # Read metadata from JSONL file
    print(f"Reading metadata from {jsonl_file_path}...")
    metadata = read_metadata_from_jsonl(jsonl_file_path)
    print(f"Found {len(metadata)} points to visualize")

    if not metadata:
        print("No data found in the file!")
        return

    # Extract connections between points
    print("Extracting connections between points...")
    connections = extract_connections(metadata)
    print(f"Found {len(connections)} unique connections")

    # Calculate center coordinates
    center_coords = calculate_center_coordinates(metadata)
    print(f"Map center: {center_coords[0]:.6f}, {center_coords[1]:.6f}")

    # Create Folium map
    print("Creating map with points and connections...")
    m = create_folium_map(metadata, center_coords, connections)

    # Determine output path
    if output_html_path is None:
        base_name = os.path.splitext(os.path.basename(jsonl_file_path))[0]
        output_html_path = f"{base_name}_map_with_connections.html"

    # Save map to HTML file
    print(f"Saving map to {output_html_path}...")
    m.save(output_html_path)
    print(f"Map saved successfully! Open {output_html_path} in your browser to view the visualization.")
    print("Use the layer control in the top-right to toggle points and connections visibility.")

    return m


if __name__ == "__main__":
    # Path to the metadata file
    root_pano_id = "DyDhU3ixcGl-9BT_SNzHTQ"
    metadata_file = f"metadata/{root_pano_id}_minimetadata.jsonl"

    # Check if file exists
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found!")
        print("Make sure you're running this script from the experimentalloading directory")
    else:
        # Create visualization
        visualize_points_on_map(metadata_file)
