import xml.etree.ElementTree as ET

def extract_polygons(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}  # define namespace
    polygons = []

    for placemark in root.findall('.//kml:Placemark', ns):  # for each placemark
        for multigeometry in placemark.findall('.//kml:MultiGeometry', ns):  # for each MultiGeometry
            for polygon in multigeometry.findall('.//kml:Polygon', ns):  # for each Polygon
                for outerboundary in polygon.findall('.//kml:outerBoundaryIs', ns):  # for each outerBoundaryIs
                    for linear_ring in outerboundary.findall('.//kml:LinearRing', ns):  # for each LinearRing
                        for coordinates in linear_ring.findall('.//kml:coordinates', ns):  # for each coordinates
                            coordinate_list = coordinates.text.strip().split(" ")
                            coordinate_tuples = [tuple(map(float, coord.split(','))) for coord in coordinate_list]
                            polygons.append(coordinate_tuples)

    return polygons  # Returns a list of polygons, each of which is a list of (longitude, latitude) pairs.

from math import sin, cos, sqrt, atan2, radians

def calculate_area(polygon):
    # This function calculates the area of a polygon on the Earth's surface,
    # given its vertices in (longitude, latitude) format. The coordinates
    # are assumed to be in degrees.

    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert coordinates to radians
    polygon = [(radians(lon), radians(lat)) for lon, lat in polygon]

    # Calculate 3D Cartesian coordinates
    cartesian_polygon = [(R*cos(lat)*cos(lon), R*cos(lat)*sin(lon), R*sin(lat)) for lon, lat in polygon]

    # Sum over edges of the polygon
    area = 0.0
    for i in range(len(cartesian_polygon)):
        j = (i + 1) % len(cartesian_polygon)
        area += cartesian_polygon[i][0] * (cartesian_polygon[j][1] - cartesian_polygon[(j + 1) % len(cartesian_polygon)][1])
    return abs(area) / 2.0

# To use the function, simply call it with a list of (longitude, latitude) pairs defining a polygon:
# polygons is the list of polygons you got from the KML file

