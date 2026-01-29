#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   QUAKE 2 MAP TO UNREAL ENGINE 5 T3D CONVERTER                            ║
║   THE ULTIMATE EDITION v2.2                                               ║
║                                                                           ║
║   Convert Quake 2 .MAP files to Unreal Engine 5 .T3D format               ║
║   with perfect geometry, proper winding, and clean actor organization     ║
║                                                                           ║
║   Created by AlleyKatPr0 — Level Design & Mapping Specialist              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        if abs(scalar) < 0.0001:
            return Vector3(0, 0, 0)
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0.0001:
            return self / length
        return Vector3(0, 0, 1)
    
    @staticmethod
    def snap_to_grid(value: float, grid_size: float) -> float:
        if grid_size <= 0:
            return value
        return round(value / grid_size) * grid_size
    
    def to_unreal(self, grid_snap: float = 2.54):
        scale = 2.54
        x = self.x * scale
        y = -self.y * scale
        z = self.z * scale
        
        if grid_snap > 0:
            x = self.snap_to_grid(x, grid_snap)
            y = self.snap_to_grid(y, grid_snap)
            z = self.snap_to_grid(z, grid_snap)
        
        return Vector3(x, y, z)
    
    def to_t3d_string(self) -> str:
        def format_component(val):
            if val >= 0:
                return f"+{abs(val):011.6f}"
            else:
                return f"-{abs(val):011.6f}"
        
        return f"{format_component(self.x)},{format_component(self.y)},{format_component(self.z)}"
    
    def is_valid(self):
        return (math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z))


class Color:
    def __init__(self, r: float, g: float, b: float, a: float = 1.0):
        self.r = max(0.0, min(1.0, r))
        self.g = max(0.0, min(1.0, g))
        self.b = max(0.0, min(1.0, b))
        self.a = max(0.0, min(1.0, a))
    
    def __repr__(self):
        return f"Color(R={self.r:.3f}, G={self.g:.3f}, B={self.b:.3f})"
    
    def to_linear_color_string(self) -> str:
        return f"(R={self.r:.6f},G={self.g:.6f},B={self.b:.6f},A={self.a:.6f})"
    
    @staticmethod
    def from_hex(hex_str: str) -> 'Color':
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return Color(r, g, b)
        return Color(1.0, 1.0, 1.0)
    
    @staticmethod
    def from_quake_value(light_value: str, color_property: Optional[str] = None) -> Tuple['Color', float]:
        parts = light_value.split()
        intensity = 300.0
        color = Color(1.0, 1.0, 1.0)
        
        if len(parts) >= 1:
            try:
                intensity = float(parts[0])
            except ValueError:
                intensity = 300.0
        
        if color_property:
            color_parts = color_property.strip().split()
            if len(color_parts) >= 3:
                try:
                    r = float(color_parts[0])
                    g = float(color_parts[1])
                    b = float(color_parts[2])
                    color = Color(r, g, b)
                    logger.debug(f"Light color from _color: {color}")
                    ue_intensity = intensity / 3.0
                    return color, ue_intensity
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse _color '{color_property}': {e}")
        
        if len(parts) >= 2 and parts[1].startswith('#'):
            color = Color.from_hex(parts[1])
            logger.debug(f"Light color from hex: {color}")
        elif len(parts) >= 4:
            try:
                r = int(parts[1]) / 255.0
                g = int(parts[2]) / 255.0
                b = int(parts[3]) / 255.0
                color = Color(r, g, b)
                logger.debug(f"Light color from RGB: {color}")
            except (ValueError, IndexError):
                pass
        
        ue_intensity = intensity / 3.0
        return color, ue_intensity


class Face:
    def __init__(self, p1: Vector3, p2: Vector3, p3: Vector3, texture: str = ""):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.texture = texture
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2).normalize()
        self.normal = normal * -1.0
        self.dist = self.normal.dot(p1)
    
    def distance_to_point(self, point: Vector3) -> float:
        return self.normal.dot(point) - self.dist


class Brush:
    def __init__(self):
        self.faces: List[Face] = []
    
    def add_face(self, face: Face):
        self.faces.append(face)
    
    def get_center(self) -> Optional[Vector3]:
        vertices = self.calculate_vertices()
        if not vertices:
            return None
        center = Vector3(0, 0, 0)
        for v in vertices:
            center = center + v
        return center / len(vertices)
    
    def calculate_vertices(self, epsilon=0.1) -> List[Vector3]:
        vertices = []
        
        if len(self.faces) < 4:
            return vertices
        
        for i in range(len(self.faces)):
            for j in range(i + 1, len(self.faces)):
                for k in range(j + 1, len(self.faces)):
                    vertex = self._intersect_three_planes(
                        self.faces[i], self.faces[j], self.faces[k]
                    )
                    
                    if vertex is None or not vertex.is_valid():
                        continue
                    
                    valid = True
                    for face in self.faces:
                        if face.distance_to_point(vertex) > epsilon:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    is_duplicate = False
                    for existing in vertices:
                        if (abs(vertex.x - existing.x) < epsilon and
                            abs(vertex.y - existing.y) < epsilon and
                            abs(vertex.z - existing.z) < epsilon):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        vertices.append(vertex)
        
        return vertices
    
    def _intersect_three_planes(self, f1: Face, f2: Face, f3: Face) -> Optional[Vector3]:
        n1, n2, n3 = f1.normal, f2.normal, f3.normal
        d1, d2, d3 = f1.dist, f2.dist, f3.dist
        
        denom = n1.dot(n2.cross(n3))
        
        if abs(denom) < 0.0001:
            return None
        
        numerator = (n2.cross(n3) * d1 + n3.cross(n1) * d2 + n1.cross(n2) * d3)
        return numerator / denom
    
    def calculate_face_polygons(self, epsilon=0.1) -> List[Dict]:
        vertices = self.calculate_vertices(epsilon)
        
        if len(vertices) < 4:
            return []
        
        polygons = []
        
        for face in self.faces:
            face_verts = []
            for vertex in vertices:
                if abs(face.distance_to_point(vertex)) < epsilon:
                    face_verts.append(vertex)
            
            if len(face_verts) < 3:
                continue
            
            center = Vector3(0, 0, 0)
            for v in face_verts:
                center = center + v
            center = center / len(face_verts)
            
            sorted_verts = self._sort_ccw(face_verts, center, face.normal)
            
            polygons.append({
                'face': face,
                'vertices': sorted_verts,
                'texture': face.texture
            })
        
        return polygons
    
    def _sort_ccw(self, vertices: List[Vector3], center: Vector3, normal: Vector3) -> List[Vector3]:
        if len(vertices) <= 2:
            return vertices
        
        if abs(normal.z) > 0.9:
            right = Vector3(1, 0, 0)
        else:
            right = Vector3(0, 0, 1)
        
        right = right - (normal * right.dot(normal))
        right = right.normalize()
        up = normal.cross(right).normalize()
        
        angles = []
        for v in vertices:
            relative = v - center
            x = relative.dot(right)
            y = relative.dot(up)
            angle = math.atan2(y, x)
            angles.append((angle, v))
        
        angles.sort(key=lambda x: x[0])
        return [v for _, v in angles]


class Entity:
    def __init__(self):
        self.properties: Dict[str, str] = {}
        self.brushes: List[Brush] = []
    
    def add_property(self, key: str, value: str):
        self.properties[key] = value
    
    def add_brush(self, brush: Brush):
        self.brushes.append(brush)
    
    def get_classname(self) -> str:
        return self.properties.get("classname", "unknown")
    
    def get_origin(self) -> Optional[Vector3]:
        origin_str = self.properties.get("origin")
        if origin_str:
            parts = origin_str.split()
            if len(parts) == 3:
                try:
                    return Vector3(float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
        
        if self.brushes:
            return self.brushes[0].get_center()
        
        return None
    
    def get_angle(self) -> float:
        angle_str = self.properties.get("angle", "0")
        try:
            return float(angle_str)
        except ValueError:
            return 0.0
    
    def is_light(self) -> bool:
        classname = self.get_classname()
        return classname.startswith("light")
    
    def is_trigger(self) -> bool:
        classname = self.get_classname()
        return classname.startswith("trigger_")
    
    def is_monster(self) -> bool:
        classname = self.get_classname()
        return classname.startswith("monster_")
    
    def get_light_properties(self) -> Tuple[Color, float, float]:
        light_value = self.properties.get("light", "300")
        color_property = self.properties.get("_color", None)
        
        color, intensity = Color.from_quake_value(light_value, color_property)
        
        attenuation_radius = math.sqrt(intensity) * 100.0
        attenuation_radius = max(100.0, min(attenuation_radius, 10000.0))
        
        if color_property:
            logger.info(f"Light with _color: {color} intensity={intensity:.1f}")
        
        return color, intensity, attenuation_radius
    
    def properties_to_string(self) -> str:
        lines = []
        lines.append(f"Quake 2 Entity: {self.get_classname()}")
        lines.append("=" * 50)
        
        for key, value in sorted(self.properties.items()):
            lines.append(f"{key}: {value}")
        
        if self.brushes:
            lines.append("")
            lines.append(f"Brushes: {len(self.brushes)}")
        
        return "\\n".join(lines)


class MAPParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entities: List[Entity] = []
    
    def parse(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"MAP file not found: {self.filepath}")
        
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        self._parse_entities(content)
    
    def _parse_entities(self, content: str):
        entity_depth = 0
        brush_depth = 0
        current_entity = None
        current_brush = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            try:
                if line == '{' and entity_depth == 0:
                    entity_depth = 1
                    current_entity = Entity()
                
                elif line == '}' and entity_depth == 1 and brush_depth == 0:
                    entity_depth = 0
                    if current_entity:
                        self.entities.append(current_entity)
                
                elif line == '{' and entity_depth == 1:
                    brush_depth = 1
                    current_brush = Brush()
                
                elif line == '}' and brush_depth == 1:
                    brush_depth = 0
                    if current_brush and current_entity:
                        current_entity.add_brush(current_brush)
                
                elif entity_depth == 1 and brush_depth == 0:
                    match = re.match(r'"([^"]+)"\s+"([^"]*)"', line)
                    if match and current_entity:
                        current_entity.add_property(match.group(1), match.group(2))
                
                elif brush_depth == 1 and current_brush:
                    face = self._parse_face(line)
                    if face:
                        current_brush.add_face(face)
            
            except Exception as e:
                logger.warning(f"Parse error on line: {line[:50]}... - {e}")
    
    def _parse_face(self, line: str) -> Optional[Face]:
        pattern = r'\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)\s*' \
                  r'\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)\s*' \
                  r'\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)\s*' \
                  r'(\S+)'
        
        match = re.match(pattern, line)
        if match:
            try:
                p1 = Vector3(float(match.group(1)), float(match.group(2)), float(match.group(3)))
                p2 = Vector3(float(match.group(4)), float(match.group(5)), float(match.group(6)))
                p3 = Vector3(float(match.group(7)), float(match.group(8)), float(match.group(9)))
                texture = match.group(10)
                
                return Face(p1, p2, p3, texture)
            except:
                return None
        
        return None


class PolygonTriangulator:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.stats = {
            'ear_clipping_success': 0,
            'fan_fallback': 0,
            'simple_fallback': 0,
            'degenerate': 0
        }
    
    def triangulate(self, vertices: List[Vector3], normal: Optional[Vector3] = None) -> List[List[Vector3]]:
        if len(vertices) < 3:
            self.stats['degenerate'] += 1
            return []
        
        if len(vertices) == 3:
            return [vertices]
        
        if normal is None:
            normal = self._calculate_normal(vertices)
            if normal is None:
                self.stats['degenerate'] += 1
                return []
        
        cleaned_vertices = self._clean_vertices(vertices, normal)
        
        if len(cleaned_vertices) < 3:
            self.stats['degenerate'] += 1
            return []
        
        if len(cleaned_vertices) == 3:
            return [cleaned_vertices]
        
        triangles = self._ear_clipping(cleaned_vertices, normal)
        
        if triangles:
            self.stats['ear_clipping_success'] += 1
            return triangles
        
        triangles = self._fan_triangulation(cleaned_vertices)
        
        if triangles:
            self.stats['fan_fallback'] += 1
            return triangles
        
        triangles = self._simple_split(cleaned_vertices)
        self.stats['simple_fallback'] += 1
        return triangles
    
    def _ear_clipping(self, vertices: List[Vector3], normal: Vector3, max_iterations: int = 1000) -> List[List[Vector3]]:
        triangles = []
        remaining = list(vertices)
        iterations = 0
        
        while len(remaining) > 3 and iterations < max_iterations:
            iterations += 1
            ear_found = False
            
            for i in range(len(remaining)):
                prev_idx = (i - 1 + len(remaining)) % len(remaining)
                next_idx = (i + 1) % len(remaining)
                
                prev = remaining[prev_idx]
                curr = remaining[i]
                next_v = remaining[next_idx]
                
                if self._is_ear(prev, curr, next_v, remaining, normal):
                    triangles.append([prev, curr, next_v])
                    remaining.pop(i)
                    ear_found = True
                    break
            
            if not ear_found:
                return []
        
        if len(remaining) == 3:
            triangles.append(remaining)
        elif len(remaining) > 3:
            return []
        
        return triangles
    
    def _is_ear(self, prev: Vector3, curr: Vector3, next_v: Vector3, 
                polygon: List[Vector3], normal: Vector3) -> bool:
        edge1 = curr - prev
        edge2 = next_v - curr
        cross = edge1.cross(edge2)
        
        dot = cross.dot(normal)
        
        if dot < -self.epsilon:
            return False
        
        if abs(dot) < self.epsilon:
            return False
        
        for p in polygon:
            if self._vec_equal(p, prev) or self._vec_equal(p, curr) or self._vec_equal(p, next_v):
                continue
            
            if self._point_in_triangle(p, prev, curr, next_v, normal):
                return False
        
        return True
    
    def _point_in_triangle(self, point: Vector3, a: Vector3, b: Vector3, 
                           c: Vector3, normal: Vector3) -> bool:
        if abs(normal.z) > 0.9:
            u_axis = Vector3(1, 0, 0)
        else:
            u_axis = Vector3(0, 0, 1)
        
        u_axis = u_axis - (normal * u_axis.dot(normal))
        u_axis = u_axis.normalize()
        v_axis = normal.cross(u_axis).normalize()
        
        def project_2d(v):
            return (v.dot(u_axis), v.dot(v_axis))
        
        p = project_2d(point)
        a2d = project_2d(a)
        b2d = project_2d(b)
        c2d = project_2d(c)
        
        v0x = c2d[0] - a2d[0]
        v0y = c2d[1] - a2d[1]
        v1x = b2d[0] - a2d[0]
        v1y = b2d[1] - a2d[1]
        v2x = p[0] - a2d[0]
        v2y = p[1] - a2d[1]
        
        dot00 = v0x * v0x + v0y * v0y
        dot01 = v0x * v1x + v0y * v1y
        dot02 = v0x * v2x + v0y * v2y
        dot11 = v1x * v1x + v1y * v1y
        dot12 = v1x * v2x + v1y * v2y
        
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < self.epsilon:
            return False
        
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        return (u > self.epsilon and 
                v > self.epsilon and 
                (u + v) < (1.0 - self.epsilon))
    
    def _fan_triangulation(self, vertices: List[Vector3]) -> List[List[Vector3]]:
        if len(vertices) < 3:
            return []
        
        if len(vertices) == 3:
            return [vertices]
        
        triangles = []
        for i in range(1, len(vertices) - 1):
            triangles.append([vertices[0], vertices[i], vertices[i + 1]])
        
        return triangles
    
    def _simple_split(self, vertices: List[Vector3]) -> List[List[Vector3]]:
        triangles = []
        for i in range(1, len(vertices) - 1):
            triangles.append([vertices[0], vertices[i], vertices[i + 1]])
        return triangles
    
    def _clean_vertices(self, vertices: List[Vector3], normal: Vector3) -> List[Vector3]:
        if len(vertices) < 3:
            return vertices
        
        cleaned = []
        
        for i in range(len(vertices)):
            prev_idx = (i - 1 + len(vertices)) % len(vertices)
            next_idx = (i + 1) % len(vertices)
            
            prev = vertices[prev_idx]
            curr = vertices[i]
            next_v = vertices[next_idx]
            
            is_duplicate = False
            for v in cleaned:
                if self._vec_equal(curr, v):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            edge1 = curr - prev
            edge2 = next_v - curr
            cross = edge1.cross(edge2)
            
            if cross.length() < self.epsilon:
                continue
            
            cleaned.append(curr)
        
        return cleaned
    
    def _calculate_normal(self, vertices: List[Vector3]) -> Optional[Vector3]:
        if len(vertices) < 3:
            return None
        
        normal = Vector3(0, 0, 0)
        
        for i in range(len(vertices)):
            curr = vertices[i]
            next_v = vertices[(i + 1) % len(vertices)]
            
            normal.x += (curr.y - next_v.y) * (curr.z + next_v.z)
            normal.y += (curr.z - next_v.z) * (curr.x + next_v.x)
            normal.z += (curr.x - next_v.x) * (curr.y + next_v.y)
        
        length = normal.length()
        if length < self.epsilon:
            return None
        
        return normal.normalize()
    
    def _vec_equal(self, a: Vector3, b: Vector3) -> bool:
        return (abs(a.x - b.x) < self.epsilon and
                abs(a.y - b.y) < self.epsilon and
                abs(a.z - b.z) < self.epsilon)
    
    def print_stats(self):
        total = sum(self.stats.values())
        if total == 0:
            return
        
        print("\n" + "="*70)
        print("TRIANGULATION STATISTICS")
        print("="*70)
        print(f"  Ear Clipping Success: {self.stats['ear_clipping_success']:6d} ({100*self.stats['ear_clipping_success']/total:.1f}%)")
        print(f"  Fan Fallback:         {self.stats['fan_fallback']:6d} ({100*self.stats['fan_fallback']/total:.1f}%)")
        print(f"  Simple Fallback:      {self.stats['simple_fallback']:6d} ({100*self.stats['simple_fallback']/total:.1f}%)")
        print(f"  Degenerate:           {self.stats['degenerate']:6d} ({100*self.stats['degenerate']/total:.1f}%)")
        print("="*70)


class T3DWriter:
    def __init__(self, output_path: str, grid_size: float = 2.54):
        self.output_path = output_path
        self.grid_size = grid_size
        self.triangulator = PolygonTriangulator(epsilon=grid_size / 10.0)
        
        self.brush_counter = 0
        self.link_counter = 0
        self.actor_counter = 0
        self.light_counter = 0
        self.trigger_counter = 0
        self.target_counter = 0
        self.monster_counter = 0
        
        self.stats = {
            'brushes': 0,
            'polygons': 0,
            'triangles': 0,
            'lights': 0,
            'lights_with_color': 0,
            'monsters': 0,
            'triggers': 0,
            'items': 0,
            'player_starts': 0
        }
    
    def write(self, entities: List[Entity]):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            self._write_header(f)
            
            for entity in entities:
                try:
                    self._write_entity(f, entity)
                except Exception as e:
                    logger.error(f"Error writing entity {entity.get_classname()}: {e}")
            
            self._write_footer(f)
    
    def _write_header(self, f):
        f.write("Begin Map\n")
        f.write("   Begin Level\n")
    
    def _write_footer(self, f):
        f.write("   End Level\n")
        self._write_surface_section(f)
        f.write("End Map\n")
    
    def _write_surface_section(self, f):
        if self.link_counter > 0:
            f.write("Begin Surface\n")
            f.write("   TEXTURE=None\n")
            f.write("   BASE      +00000.000000,+00000.000000,+00000.000000\n")
            f.write("   TEXTUREU  +00000.000000,+00001.000000,+00000.000000\n")
            f.write("   TEXTUREV  +00000.000000,+00000.000000,-00001.000000\n")
            f.write("   NORMAL    -00001.000000,+00000.000000,+00000.000000\n")
            f.write("   POLYFLAGS=33554432\n")
            f.write("End Surface\n")
    
    def _write_entity(self, f, entity: Entity):
        classname = entity.get_classname()
        
        if entity.is_light():
            self._write_light_actor(f, entity)
            self.stats['lights'] += 1
            if '_color' in entity.properties:
                self.stats['lights_with_color'] += 1
        
        elif entity.is_monster():
            self._write_monster_actor(f, entity)
            self.stats['monsters'] += 1
        
        elif entity.is_trigger():
            self._write_target_point(f, entity)
            self.stats['triggers'] += 1
        
        elif classname.startswith("target_"):
            self._write_target_point(f, entity)
        
        elif (classname.startswith("item_") or 
              classname.startswith("ammo_") or 
              classname.startswith("weapon_")):
            self._write_note_actor(f, entity)
            self.stats['items'] += 1
        
        elif classname.startswith("info_player"):
            self._write_player_start(f, entity)
            self.stats['player_starts'] += 1
        
        elif classname == "worldspawn":
            for brush in entity.brushes:
                self._write_brush_actor(f, brush)
                self.stats['brushes'] += 1
        
        elif entity.brushes:
            for brush in entity.brushes:
                self._write_brush_actor(f, brush)
                self.stats['brushes'] += 1
        
        else:
            if classname not in ["info_null", "func_group"]:
                self._write_note_actor(f, entity)
    
    def _write_light_actor(self, f, entity: Entity):
        origin = entity.get_origin()
        if not origin:
            logger.warning("Light has no origin")
            return
        
        origin_unreal = origin.to_unreal(grid_snap=self.grid_size)
        color, intensity, attenuation_radius = entity.get_light_properties()
        
        self.light_counter += 1
        light_name = f"Light_{self.light_counter}"
        
        f.write(f"      Begin Actor Class=/Script/Engine.PointLight Name={light_name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.PointLightComponent Name=\"LightComponent0\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"LightComponent0\"\n")
        f.write(f"            IntensityUnits=Candelas\n")
        f.write(f"            Intensity={intensity:.6f}\n")
        f.write(f"            LightColor={color.to_linear_color_string()}\n")
        f.write(f"            AttenuationRadius={attenuation_radius:.6f}\n")
        f.write(f"            bUseTemperature=False\n")
        f.write(f"            CastShadows=True\n")
        f.write(f"            RelativeLocation=(X={origin_unreal.x:.6f},Y={origin_unreal.y:.6f},Z={origin_unreal.z:.6f})\n")
        f.write(f"         End Object\n")
        f.write(f"         PointLightComponent=\"LightComponent0\"\n")
        f.write(f"         LightComponent=\"LightComponent0\"\n")
        f.write(f"         RootComponent=\"LightComponent0\"\n")
        f.write(f"         ActorLabel=\"{light_name}\"\n")
        f.write(f"         FolderPath=\"Quake2/Lights\"\n")
        f.write(f"      End Actor\n")
    
    def _write_monster_actor(self, f, entity: Entity):
        origin = entity.get_origin()
        if not origin:
            return
        
        origin_unreal = origin.to_unreal(grid_snap=self.grid_size)
        classname = entity.get_classname()
        angle = entity.get_angle()
        
        self.monster_counter += 1
        monster_type = classname.replace("monster_", "").replace("_", " ").title().replace(" ", "")
        name = f"{monster_type}_{self.monster_counter}"
        
        f.write(f"      Begin Actor Class=/Script/Engine.Actor Name={name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.SceneComponent Name=\"DefaultSceneRoot\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"DefaultSceneRoot\"\n")
        f.write(f"            RelativeLocation=(X={origin_unreal.x:.6f},Y={origin_unreal.y:.6f},Z={origin_unreal.z:.6f})\n")
        
        if angle != 0.0:
            unreal_yaw = int((angle / 360.0) * 65536) % 65536
            f.write(f"            RelativeRotation=(Pitch=0,Yaw={unreal_yaw},Roll=0)\n")
        
        f.write(f"         End Object\n")
        f.write(f"         RootComponent=\"DefaultSceneRoot\"\n")
        f.write(f"         ActorLabel=\"{monster_type}_{self.monster_counter}\"\n")
        f.write(f"         Tags=\"Q2Monster\",\"{classname}\"\n")
        f.write(f"         FolderPath=\"Quake2/Monsters\"\n")
        f.write(f"      End Actor\n")
    
    def _write_target_point(self, f, entity: Entity):
        origin = entity.get_origin()
        if not origin:
            return
        
        origin_unreal = origin.to_unreal(grid_snap=self.grid_size)
        classname = entity.get_classname()
        
        self.target_counter += 1
        clean_name = classname.replace("trigger_", "Trigger").replace("target_", "Target").replace("_", " ").title().replace(" ", "")
        name = f"{clean_name}_{self.target_counter}"
        
        f.write(f"      Begin Actor Class=/Script/Engine.TargetPoint Name={name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.SceneComponent Name=\"SceneComp\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.BillboardComponent Name=\"Sprite\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.ArrowComponent Name=\"Arrow\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"SceneComp\"\n")
        f.write(f"            RelativeLocation=(X={origin_unreal.x:.6f},Y={origin_unreal.y:.6f},Z={origin_unreal.z:.6f})\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Sprite\"\n")
        f.write(f"            AttachParent=\"SceneComp\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Arrow\"\n")
        f.write(f"            AttachParent=\"Sprite\"\n")
        f.write(f"         End Object\n")
        f.write(f"         SpriteComponent=\"Sprite\"\n")
        f.write(f"         ArrowComponent=\"Arrow\"\n")
        f.write(f"         RootComponent=\"SceneComp\"\n")
        f.write(f"         ActorLabel=\"{clean_name}_{self.target_counter}\"\n")
        f.write(f"         Tags=\"Q2Target\",\"{classname}\"\n")
        f.write(f"         FolderPath=\"Quake2/Triggers\"\n")
        f.write(f"      End Actor\n")
    
    def _write_player_start(self, f, entity: Entity):
        origin = entity.get_origin()
        if not origin:
            return
        
        origin_unreal = origin.to_unreal(grid_snap=self.grid_size)
        angle = entity.get_angle()
        
        self.actor_counter += 1
        name = f"PlayerStart_{self.actor_counter}"
        
        f.write(f"      Begin Actor Class=/Script/Engine.PlayerStart Name={name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.CapsuleComponent Name=\"CollisionCapsule\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.BillboardComponent Name=\"Sprite\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.BillboardComponent Name=\"Sprite2\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.ArrowComponent Name=\"Arrow\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"CollisionCapsule\"\n")
        f.write(f"            AreaClass=\"/Script/CoreUObject.Class'/Script/NavigationSystem.NavArea_Obstacle'\"\n")
        f.write(f"            RelativeLocation=(X={origin_unreal.x:.6f},Y={origin_unreal.y:.6f},Z={origin_unreal.z:.6f})\n")
        
        if angle != 0.0:
            unreal_yaw = int((angle / 360.0) * 65536) % 65536
            f.write(f"            RelativeRotation=(Pitch=0,Yaw={unreal_yaw},Roll=0)\n")
        
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Sprite\"\n")
        f.write(f"            AttachParent=\"CollisionCapsule\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Sprite2\"\n")
        f.write(f"            AttachParent=\"CollisionCapsule\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Arrow\"\n")
        f.write(f"            AttachParent=\"CollisionCapsule\"\n")
        f.write(f"         End Object\n")
        f.write(f"         ArrowComponent=\"Arrow\"\n")
        f.write(f"         CapsuleComponent=\"CollisionCapsule\"\n")
        f.write(f"         GoodSprite=\"Sprite\"\n")
        f.write(f"         BadSprite=\"Sprite2\"\n")
        f.write(f"         RootComponent=\"CollisionCapsule\"\n")
        f.write(f"         ActorLabel=\"{name}\"\n")
        f.write(f"         FolderPath=\"Quake2/PlayerStarts\"\n")
        f.write(f"      End Actor\n")
    
    def _write_note_actor(self, f, entity: Entity):
        classname = entity.get_classname()
        origin = entity.get_origin()
        
        if not origin:
            return
        
        origin_unreal = origin.to_unreal(grid_snap=self.grid_size)
        properties_text = entity.properties_to_string()
        
        self.actor_counter += 1
        clean_type = classname.replace("item_", "").replace("ammo_", "").replace("weapon_", "").replace("_", " ").title().replace(" ", "")
        name = f"{clean_type}_{self.actor_counter}"
        
        if classname.startswith("item_"):
            folder = "Quake2/Items"
        elif classname.startswith("ammo_"):
            folder = "Quake2/Ammo"
        elif classname.startswith("weapon_"):
            folder = "Quake2/Weapons"
        else:
            folder = "Quake2/Other"
        
        f.write(f"      Begin Actor Class=/Script/Engine.Note Name={name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.SceneComponent Name=\"SceneComp\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.ArrowComponent Name=\"Arrow\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Class=/Script/Engine.BillboardComponent Name=\"Sprite\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"SceneComp\"\n")
        f.write(f"            RelativeLocation=(X={origin_unreal.x:.6f},Y={origin_unreal.y:.6f},Z={origin_unreal.z:.6f})\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Arrow\"\n")
        f.write(f"            AttachParent=\"SceneComp\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"Sprite\"\n")
        f.write(f"            AttachParent=\"SceneComp\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Text=\"{properties_text}\"\n")
        f.write(f"         SpriteComponent=\"Sprite\"\n")
        f.write(f"         ArrowComponent=\"Arrow\"\n")
        f.write(f"         RootComponent=\"SceneComp\"\n")
        f.write(f"         ActorLabel=\"{clean_type}_{self.actor_counter}\"\n")
        f.write(f"         Tags=\"Q2Item\",\"{classname}\"\n")
        f.write(f"         FolderPath=\"{folder}\"\n")
        f.write(f"      End Actor\n")
    
    def _write_brush_actor(self, f, brush: Brush):
        polygons = brush.calculate_face_polygons()
        
        if not polygons:
            return
        
        self.brush_counter += 1
        brush_name = f"Brush_{self.brush_counter}"
        model_name = f"Model_{self.brush_counter}"
        
        # Write Brush Actor with proper UE5 structure
        f.write(f"      Begin Actor Class=/Script/Engine.Brush Name={brush_name}\n")
        f.write(f"         Begin Object Class=/Script/Engine.BrushComponent Name=\"BrushComponent0\"\n")
        f.write(f"         End Object\n")
        f.write(f"         Begin Object Name=\"BrushComponent0\"\n")
        f.write(f"            Brush=\"/Script/Engine.Model'{model_name}'\"\n")
        f.write(f"         End Object\n")
        f.write(f"         BrushType=Brush_Add\n")
        
        # CRITICAL: Write the Brush definition WITH Name parameter
        f.write(f"         Begin Brush Name={model_name}\n")
        f.write(f"            Begin PolyList\n")
        
        # Write all polygons
        for polygon in polygons:
            self._write_polygon(f, polygon)
            self.stats['polygons'] += 1
        
        f.write(f"            End PolyList\n")
        f.write(f"         End Brush\n")
        
        # Set properties AFTER the Brush definition
        f.write(f"         Brush=\"/Script/Engine.Model'{model_name}'\"\n")
        f.write(f"         BrushComponent=\"BrushComponent0\"\n")
        f.write(f"         SpawnCollisionHandlingMethod=AlwaysSpawn\n")
        f.write(f"         RootComponent=\"BrushComponent0\"\n")
        f.write(f"         ActorLabel=\"{brush_name}\"\n")
        f.write(f"         FolderPath=\"Quake2/Geometry\"\n")
        
        f.write(f"      End Actor\n")
    
    def _write_polygon(self, f, polygon: Dict):
        face = polygon['face']
        vertices = polygon['vertices']
        
        # CRITICAL: Must have at least 3 vertices
        if len(vertices) < 3:
            logger.warning(f"Skipping polygon with {len(vertices)} vertices")
            return
        
        unreal_verts = [v.to_unreal(grid_snap=self.grid_size) for v in vertices]
        
        # Validate all vertices are valid
        valid_verts = [v for v in unreal_verts if v.is_valid()]
        if len(valid_verts) < 3:
            logger.warning(f"Skipping polygon with invalid vertices")
            return
        
        unreal_normal = face.normal.to_unreal(grid_snap=0).normalize()
        
        # Validate normal
        if not unreal_normal.is_valid() or unreal_normal.length() < 0.001:
            logger.warning(f"Skipping polygon with invalid normal")
            return
        
        # Fix winding
        unreal_verts.reverse()
        unreal_normal = unreal_normal * -1.0
        
        # Calculate texture vectors
        if abs(unreal_normal.z) > 0.9:
            texture_u = Vector3(1, 0, 0)
        else:
            texture_u = Vector3(0, 0, 1).cross(unreal_normal).normalize()
        
        texture_v = unreal_normal.cross(texture_u).normalize()
        
        # Triangulate
        triangles = self.triangulator.triangulate(unreal_verts, unreal_normal)
        
        # CRITICAL: Skip if triangulation failed
        if not triangles:
            logger.warning(f"Triangulation failed for polygon with {len(unreal_verts)} vertices")
            return
        
        # Write each triangle
        for tri in triangles:
            # CRITICAL: Validate triangle has exactly 3 vertices
            if len(tri) != 3:
                logger.warning(f"Skipping invalid triangle with {len(tri)} vertices")
                continue
            
            # Validate no duplicate vertices in triangle
            if (self.triangulator._vec_equal(tri[0], tri[1]) or 
                self.triangulator._vec_equal(tri[1], tri[2]) or 
                self.triangulator._vec_equal(tri[0], tri[2])):
                logger.warning(f"Skipping degenerate triangle with duplicate vertices")
                continue
            
            self.link_counter += 1
            self.stats['triangles'] += 1
            
            # CRITICAL: Include Link parameter (this is required by UE5!)
            f.write(f"               Begin Polygon Link={self.link_counter}\n")
            f.write(f"                  Origin   {tri[0].to_t3d_string()}\n")
            f.write(f"                  Normal   {unreal_normal.to_t3d_string()}\n")
            f.write(f"                  TextureU {texture_u.to_t3d_string()}\n")
            f.write(f"                  TextureV {texture_v.to_t3d_string()}\n")
            
            for vertex in tri:
                f.write(f"                  Vertex   {vertex.to_t3d_string()}\n")
            
            f.write(f"               End Polygon\n")
    
    def print_stats(self):
        print("\n" + "="*70)
        print("CONVERSION STATISTICS")
        print("="*70)
        print(f"  Brushes:       {self.stats['brushes']:6d}")
        print(f"  Polygons:      {self.stats['polygons']:6d}")
        print(f"  Triangles:     {self.stats['triangles']:6d}")
        print(f"  PointLights:   {self.stats['lights']:6d}")
        
        if self.stats['lights'] > 0:
            color_pct = (self.stats['lights_with_color'] / self.stats['lights']) * 100
            print(f"    with _color: {self.stats['lights_with_color']:6d} ({color_pct:.1f}%)")
        
        print(f"  Monsters:      {self.stats['monsters']:6d}")
        print(f"  Triggers:      {self.stats['triggers']:6d}")
        print(f"  Items/Ammo:    {self.stats['items']:6d}")
        print(f"  PlayerStarts:  {self.stats['player_starts']:6d}")
        print("="*70)
        
        self.triangulator.print_stats()


def convert_map_to_t3d(map_filepath: str, t3d_filepath: str, grid_size: float = 2.54):
    logger.info(f"╔{'═'*68}╗")
    logger.info(f"║{'QUAKE 2 → UNREAL ENGINE 5 CONVERTER v2.2':^68}║")
    logger.info(f"╚{'═'*68}╝")
    logger.info(f"")
    logger.info(f"Input:  {map_filepath}")
    logger.info(f"Output: {t3d_filepath}")
    logger.info(f"Grid:   {grid_size}cm snapping")
    logger.info(f"")
    
    logger.info("Parsing MAP file...")
    parser = MAPParser(map_filepath)
    parser.parse()
    
    total_brushes = sum(len(e.brushes) for e in parser.entities)
    total_lights = sum(1 for e in parser.entities if e.is_light())
    total_monsters = sum(1 for e in parser.entities if e.is_monster())
    total_triggers = sum(1 for e in parser.entities if e.is_trigger())
    
    logger.info(f"Found:")
    logger.info(f"  • {len(parser.entities)} entities")
    logger.info(f"  • {total_brushes} brushes")
    logger.info(f"  • {total_lights} lights")
    logger.info(f"  • {total_monsters} monsters")
    logger.info(f"  • {total_triggers} triggers")
    logger.info(f"")
    
    logger.info("Writing T3D file...")
    writer = T3DWriter(t3d_filepath, grid_size=grid_size)
    writer.write(parser.entities)
    
    writer.print_stats()
    
    logger.info(f"\n✓ Conversion complete!")
    logger.info(f"✓ Output: {t3d_filepath}")


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ██╗   ██╗ █████╗ ██╗  ██╗███████╗    ██████╗                  ║
║  ██╔═══██╗██║   ██║██╔══██╗██║ ██╔╝██╔════╝    ╚════██╗                 ║
║  ██║   ██║██║   ██║███████║█████╔╝ █████╗       █████╔╝                 ║
║  ██║▄▄ ██║██║   ██║██╔══██║██╔═██╗ ██╔══╝      ██╔═══╝                  ║
║  ╚██████╔╝╚██████╔╝██║  ██║██║  ██╗███████╗    ███████╗                 ║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝                 ║
║                                                                           ║
║                          → UNREAL ENGINE 5                                ║
║                            T3D CONVERTER                                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()
    
    if len(sys.argv) < 2:
        print("Drag a .map file onto this script, or run:")
        print(f"  python {sys.argv[0]} <input.map> [output.t3d] [grid_size]")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"\n❌ ERROR: File not found: {input_file}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    output_file = sys.argv[2] if len(sys.argv) >= 3 else str(Path(input_file).with_suffix('.t3d'))
    grid_size = float(sys.argv[3]) if len(sys.argv) >= 4 else 2.54
    
    try:
        convert_map_to_t3d(input_file, output_file, grid_size)
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS!")
        print(f"{'='*70}")
        print(f"\nOutput: {output_file}")
        print(f"\nTo import into UE5:")
        print(f"  1. Open the .t3d file in a text editor")
        print(f"  2. Select all (Ctrl+A) and copy (Ctrl+C)")
        print(f"  3. Paste (Ctrl+V) into UE5 viewport")
        print(f"\n{'='*70}")
        
        input("\nPress Enter to exit...")
        
    except Exception as e:
        logger.error(f"\n❌ CONVERSION FAILED!")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
