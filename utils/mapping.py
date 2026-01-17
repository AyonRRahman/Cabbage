import folium
from folium.plugins import FastMarkerCluster
import re

def create_cluster_map(cabbage_positions, MAP_OUTPUT_PATH: str):
    '''
    Cabbage Cluster Map.
    Args:
        cabbage_positions 
        MAP_OUTPUT_PATH (str): The file path to save the map.
    '''
    if cabbage_positions:
        lats = [k[0] for k in cabbage_positions]
        lons = [k[1] for k in cabbage_positions]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=21,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23
        )
        marker_data = []
        for i, ((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
            print(f"Adding marker {i+1}/{len(cabbage_positions)}", end="\r")
            max_conf = max(conf for conf, _ in records)
            frames_seen = sorted(set(frame for _, frame in records))
            popup_text = (
                f"<b>Cabbage</b><br>"
                f"Max Conf: {max_conf:.2f}<br>"
                f"Seen in {len(frames_seen)} frames<br>"
                f"First: {frames_seen[0]}, Last: {frames_seen[-1]}"
            )
            marker_data.append([cab_lat, cab_lon, popup_text])

        callback = """(
            function(row) {
                var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
                    radius: 1.5,
                    color: '#ff0000',
                    weight: 1,
                    fillColor: '#ff3333',
                    fillOpacity: 0.9
                });
                marker.bindPopup(row[2]);
                return marker;
            }
        )"""
        FastMarkerCluster(data=marker_data, callback=callback).add_to(m)

        m.save(MAP_OUTPUT_PATH)
        
        # Minification (optional)
        with open(MAP_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            html = f.read()
        html = re.sub(r'([0-9a-f]{8})[0-9a-f-]{12,}', r'\1', html)
        lines = [line.rstrip() for line in html.split('\n')]
        with open(MAP_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Map saved: {MAP_OUTPUT_PATH}")
        print(f"Total unique cabbages mapped: {len(cabbage_positions)}")
    else:
        print("No cabbages detected.")


def create_map(cabbage_positions, MAP_OUTPUT_PATH: str):
    '''
    Cabbage individual Map.
    Args:
        cabbage_positions 
        MAP_OUTPUT_PATH (str): The file path to save the map.
    '''
    if cabbage_positions:
        lats = [k[0] for k in cabbage_positions]
        lons = [k[1] for k in cabbage_positions]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=21,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            max_zoom=23
        )
        for i, ((cab_lat, cab_lon), records) in enumerate(cabbage_positions.items()):
            print(f"Adding marker {i+1}/{len(cabbage_positions)}", end="\r")
            max_conf = max(conf for conf, _ in records)
            frames_seen = sorted(set(frame for _, frame in records))
            popup_text = f"<b>Cabbage</b><br>Max Conf: {max_conf:.2f}<br>Seen in {len(frames_seen)} frames<br>First: {frames_seen[0]}, Last: {frames_seen[-1]}"
            folium.CircleMarker(
                location=[cab_lat, cab_lon],
                radius=1.5,
                color="#ff0000",
                weight=1,
                fill=True,
                fill_color="#ff3333",
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

        m.save(MAP_OUTPUT_PATH)
        print(f"Map saved: {MAP_OUTPUT_PATH}")
        print(f"Total unique cabbages mapped: {len(cabbage_positions)}")
    else:
        print("No cabbages detected.")


