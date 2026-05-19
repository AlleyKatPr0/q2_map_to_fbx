# Quake II Map to FBX

Convert Quake II-style `.map` files into `.fbx` scenes for use in modern 3D tools and game engines.

This project exports brush-based map geometry and texture coordinates to FBX, making it easier to inspect or reuse classic level data in software such as Blender, Maya, or any other FBX-compatible tool.

## Features

- Export brush geometry as polygonal meshes
- Export texture coordinates
- Generate a complete FBX scene from a Quake II `.map` file

## Current Limitations

- Internal or non-visible faces are not yet removed
- Requires access to the original texture assets
- Support may vary depending on map editor output and texture archive layout

## Requirements

- Python
- Autodesk FBX SDK and Python FBX SDK
- Pillow 2.8.1 or newer
- A QuakeEd4-compatible `.map` file
- Texture assets from the relevant game archive

Supported editors may include:

- Embrace
- QE4
- QERadiant
- WorldCraft

## Installation

1. Install Python.
2. Install Pillow:

   ```bash
   pip install pillow
   ```

3. Install the Autodesk FBX SDK and Python bindings.
4. Clone this repository:

   ```bash
   git clone https://github.com/AlleyKatPr0/q2_map_to_fbx.git
   cd q2_map_to_fbx
   ```

## Usage

Run the converter with a source map file and output FBX path.

```bash
python main.py input.map output.fbx
```

> Update the command above if this repository uses a different script name or CLI entry point.

## Example Output

Example export: Prague North Quarter from *Vampire: The Masquerade – Redemption*.

![Example export](http://s22.postimg.org/y464wfy01/north_district_night.jpg)

## How It Works

This project was built using the Quake II QE4 source code as a reference for parsing map geometry and exporting mesh data correctly.

FBX output is generated with the Autodesk Python FBX SDK.

## Roadmap

- Remove faces that are not visible from inside the playable space
- Improve texture and material handling
- Add better validation and error reporting

## Contributing

Issues and pull requests are welcome.

## License

Add a license section here if the repository includes one.
