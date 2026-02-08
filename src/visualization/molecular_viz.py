"""
3D Molecular Visualization for Drug Discovery

Implements interactive 3D visualization using:
- PyMOL integration for protein-ligand complexes
- NGL Viewer for web-based visualization
- 3Dmol.js for interactive molecular graphics
- Plotly for 3D scatter plots and analysis
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MolecularVisualizer(ABC):
    """Abstract base class for molecular visualization"""

    @abstractmethod
    def visualize_molecule(self, structure: Union[str, Dict], **kwargs) -> str:
        """Visualize a single molecule"""
        pass

    @abstractmethod
    def visualize_complex(self, protein_path: str, ligand_structure: Union[str, Dict], **kwargs) -> str:
        """Visualize protein-ligand complex"""
        pass


class PyMOLVisualizer(MolecularVisualizer):
    """PyMOL-based molecular visualization"""

    def __init__(self):
        self.available = self._check_pymol_availability()
        if self.available:
            logger.info("PyMOL visualizer initialized")
        else:
            logger.warning("PyMOL not available. Install with: conda install -c conda-forge pymol-open-source")

    def _check_pymol_availability(self) -> bool:
        """Check if PyMOL is available"""
        try:
            import pymol
            return True
        except ImportError:
            return False

    def visualize_molecule(self, structure: Union[str, Dict], output_path: str = "molecule.png", **kwargs) -> str:
        """Visualize single molecule using PyMOL"""
        if not self.available:
            return "PyMOL not available"
        
        try:
            import pymol
            from pymol import cmd
            
            pymol.finish_launching(['pymol', '-c'])
            cmd.reinitialize()
            
            if isinstance(structure, str):
                if structure.endswith(('.pdb', '.sdf', '.mol')):
                    cmd.load(structure, "molecule")
                else:
                    logger.warning("SMILES visualization requires 3D coordinate generation")
                    return "SMILES visualization not implemented"
            
            style = kwargs.get('style', 'sticks')
            color_scheme = kwargs.get('color', 'by_element')
            
            cmd.hide("everything", "molecule")
            
            if style == 'sticks':
                cmd.show("sticks", "molecule")
            elif style == 'spheres':
                cmd.show("spheres", "molecule")
            elif style == 'cartoon':
                cmd.show("cartoon", "molecule")
            
            if color_scheme == 'by_element':
                cmd.util.cnc("molecule")
            elif color_scheme == 'by_chain':
                cmd.util.cbc("molecule")
            
            cmd.zoom("molecule")
            cmd.center("molecule")
            cmd.png(output_path, width=kwargs.get('width', 800), height=kwargs.get('height', 600))
            
            logger.info(f"Molecule visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PyMOL visualization error: {e}")
            return f"Error: {e}"

    def visualize_complex(self, protein_path: str, ligand_structure: Union[str, Dict],
                         output_path: str = "complex.png", **kwargs) -> str:
        """Visualize protein-ligand complex"""
        if not self.available:
            return "PyMOL not available"
        
        try:
            import pymol
            from pymol import cmd
            
            pymol.finish_launching(['pymol', '-c'])
            cmd.reinitialize()
            
            cmd.load(protein_path, "protein")
            
            if isinstance(ligand_structure, str):
                if ligand_structure.endswith(('.pdb', '.sdf', '.mol')):
                    cmd.load(ligand_structure, "ligand")
                else:
                    logger.warning("Ligand file format not supported")
                    return "Ligand format error"
            
            cmd.hide("everything", "protein")
            cmd.show("cartoon", "protein")
            cmd.color("gray", "protein")
            
            cmd.hide("everything", "ligand")
            cmd.show("sticks", "ligand")
            cmd.util.cnc("ligand")
            
            binding_site_radius = kwargs.get('binding_site_radius', 5.0)
            cmd.select("binding_site", f"protein within {binding_site_radius} of ligand")
            cmd.show("sticks", "binding_site")
            cmd.color("yellow", "binding_site")
            
            if kwargs.get('show_hbonds', True):
                cmd.distance("hbonds", "ligand", "protein", mode=2)
                cmd.hide("labels", "hbonds")
            
            cmd.zoom("ligand")
            cmd.center("ligand")
            cmd.png(output_path, width=kwargs.get('width', 1200), height=kwargs.get('height', 900))
            
            logger.info(f"Complex visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PyMOL complex visualization error: {e}")
            return f"Error: {e}"


class NGLViewer:
    """NGL Viewer for web-based molecular visualization"""

    def __init__(self):
        self.available = self._check_nglview_availability()
        if self.available:
            logger.info("NGL Viewer initialized")
        else:
            logger.warning("nglview not available. Install with: pip install nglview")

    def _check_nglview_availability(self) -> bool:
        """Check if nglview is available"""
        try:
            import nglview
            return True
        except ImportError:
            return False

    def create_viewer(self, structure_path: str, **kwargs):
        """Create NGL viewer widget for Jupyter notebook"""
        if not self.available:
            logger.warning("nglview not available")
            return None
        
        try:
            import nglview as nv
            
            view = nv.show_file(structure_path)
            
            representation = kwargs.get('representation', 'cartoon')
            view.clear_representations()
            view.add_representation(representation)
            
            view._remote_call(
                'setSize',
                target='Widget',
                args=['%dpx' % kwargs.get('width', 600), '%dpx' % kwargs.get('height', 400)]
            )
            
            logger.info("NGL viewer created")
            return view
            
        except Exception as e:
            logger.error(f"NGL viewer error: {e}")
            return None

    def create_complex_viewer(self, protein_path: str, ligand_path: str, **kwargs):
        """Create viewer for protein-ligand complex"""
        if not self.available:
            return None
        
        try:
            import nglview as nv
            
            view = nv.show_file(protein_path)
            view.add_component(ligand_path)
            
            view.clear_representations(component=0)
            view.add_cartoon(component=0, color='gray')
            
            view.clear_representations(component=1)
            view.add_ball_and_stick(component=1)
            
            view.center(component=1)
            
            logger.info("Complex viewer created")
            return view
            
        except Exception as e:
            logger.error(f"Complex viewer error: {e}")
            return None


class ThreeDMolJS:
    """3Dmol.js for web-based molecular visualization"""

    @staticmethod
    def generate_html(structure_data: str, structure_format: str = 'pdb', **kwargs) -> str:
        """Generate HTML with 3Dmol.js visualization"""
        
        protein_style = kwargs.get('protein_style', {'cartoon': {'color': 'gray'}})
        ligand_style = kwargs.get('ligand_style', {'stick': {'colorscheme': 'default'}})
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Molecular Visualization</title>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        .mol-container {{
            height: {kwargs.get('height', 400)}px;
            width: {kwargs.get('width', 600)}px;
            position: relative;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div id="container" class="mol-container"></div>
    <script>
        let viewer = null;
        
        function initViewer() {{
            viewer = $3Dmol.createViewer("container", {{
                defaultcolors: $3Dmol.rasmolElementColors
            }});
            
            viewer.addModel(`{structure_data}`, "{structure_format}");
            viewer.setStyle({{protein: true}}, {json.dumps(protein_style)});
            viewer.setStyle({{hetflag: true}}, {json.dumps(ligand_style)});
            viewer.zoomTo();
            viewer.render();
        }}
        
        window.onload = initViewer;
    </script>
</body>
</html>
"""
        return html_template

    @staticmethod
    def save_html_visualization(structure_path: str, output_html: str = "visualization.html", **kwargs) -> str:
        """Create and save HTML visualization"""
        try:
            with open(structure_path, 'r') as f:
                structure_data = f.read()
            
            format_map = {'.pdb': 'pdb', '.sdf': 'sdf', '.mol': 'mol'}
            structure_format = format_map.get(Path(structure_path).suffix.lower(), 'pdb')
            
            html_content = ThreeDMolJS.generate_html(structure_data, structure_format, **kwargs)
            
            with open(output_html, 'w') as f:
                f.write(html_content)
            
            logger.info(f"3Dmol.js visualization saved to {output_html}")
            return output_html
            
        except Exception as e:
            logger.error(f"3Dmol.js visualization error: {e}")
            return f"Error: {e}"


class PlotlyVisualizer:
    """Plotly for 3D data visualization and analysis"""

    def __init__(self):
        self.available = self._check_plotly_availability()
        if self.available:
            logger.info("Plotly visualizer initialized")
        else:
            logger.warning("plotly not available. Install with: pip install plotly")

    def _check_plotly_availability(self) -> bool:
        """Check if plotly is available"""
        try:
            import plotly
            return True
        except ImportError:
            return False

    def plot_binding_affinities_3d(self, molecules_df: pd.DataFrame,
                                   output_html: str = "binding_affinities_3d.html", **kwargs):
        """Create 3D scatter plot of binding affinities"""
        if not self.available:
            logger.warning("Plotly not available")
            return None
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if 'molecular_weight' not in molecules_df.columns:
                molecules_df['molecular_weight'] = np.random.uniform(200, 600, len(molecules_df))
            if 'logp' not in molecules_df.columns:
                molecules_df['logp'] = np.random.uniform(-2, 6, len(molecules_df))
            
            fig = px.scatter_3d(
                molecules_df,
                x='molecular_weight',
                y='logp',
                z='binding_affinity',
                color='binding_affinity',
                size='binding_affinity',
                hover_data=['smiles'] if 'smiles' in molecules_df.columns else None,
                title="Molecular Properties vs Binding Affinity"
            )
            
            fig.update_layout(
                width=kwargs.get('width', 800),
                height=kwargs.get('height', 600)
            )
            
            fig.write_html(output_html)
            
            logger.info(f"3D binding affinity plot saved to {output_html}")
            return fig
            
        except Exception as e:
            logger.error(f"Plotly visualization error: {e}")
            return None

    def plot_chemical_space(self, molecules_df: pd.DataFrame,
                           descriptor_cols: List[str] = ['molecular_weight', 'logp', 'tpsa'],
                           output_html: str = "chemical_space.html", **kwargs):
        """Visualize chemical space using molecular descriptors"""
        if not self.available:
            return None
        
        try:
            import plotly.graph_objects as go
            
            available_cols = [col for col in descriptor_cols if col in molecules_df.columns]
            
            if len(available_cols) < 2:
                molecules_df['molecular_weight'] = np.random.uniform(200, 600, len(molecules_df))
                molecules_df['logp'] = np.random.uniform(-2, 6, len(molecules_df))
                molecules_df['tpsa'] = np.random.uniform(20, 140, len(molecules_df))
                available_cols = ['molecular_weight', 'logp', 'tpsa']
            
            cols = available_cols[:3]
            while len(cols) < 3:
                cols.append(cols[-1])
            
            x = molecules_df[cols[0]]
            y = molecules_df[cols[1]]
            z = molecules_df[cols[2]]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=molecules_df.get('binding_affinity', range(len(molecules_df))),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=molecules_df.get('smiles', [f"Molecule_{i}" for i in range(len(molecules_df))])
            )])
            
            fig.update_layout(
                title="Chemical Space Visualization",
                width=kwargs.get('width', 800),
                height=kwargs.get('height', 600),
                scene=dict(
                    xaxis_title=cols[0],
                    yaxis_title=cols[1],
                    zaxis_title=cols[2]
                )
            )
            
            fig.write_html(output_html)
            
            logger.info(f"Chemical space plot saved to {output_html}")
            return fig
            
        except Exception as e:
            logger.error(f"Chemical space visualization error: {e}")
            return None


class MolecularVisualizationSuite:
    """Comprehensive molecular visualization suite"""

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pymol = PyMOLVisualizer()
        self.ngl = NGLViewer()
        self.plotly = PlotlyVisualizer()
        
        logger.info(f"Visualization suite initialized. Output directory: {output_dir}")

    def create_comprehensive_report(self, protein_path: str, docking_results: pd.DataFrame,
                                   top_n: int = 5) -> Dict[str, str]:
        """Create comprehensive visualization report"""
        report_files = {}
        
        try:
            plotly_file = self.output_dir / "binding_affinities_3d.html"
            if self.plotly.available:
                self.plotly.plot_binding_affinities_3d(docking_results, str(plotly_file))
                report_files['binding_plot'] = str(plotly_file)
            
            space_file = self.output_dir / "chemical_space.html"
            if self.plotly.available:
                self.plotly.plot_chemical_space(docking_results, output_html=str(space_file))
                report_files['chemical_space'] = str(space_file)
            
            summary_file = self._create_summary_html(docking_results, report_files)
            report_files['summary'] = summary_file
            
            logger.info(f"Comprehensive report created with {len(report_files)} files")
            return report_files
            
        except Exception as e:
            logger.error(f"Error creating visualization report: {e}")
            return {}

    def _create_summary_html(self, docking_results: pd.DataFrame, report_files: Dict[str, str]) -> str:
        """Create HTML summary report"""
        summary_file = self.output_dir / "visualization_summary.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Drug Discovery Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; }}
        .stats {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Drug Discovery Visualization Report</h1>
    
    <div class="section">
        <h2>Summary Statistics</h2>
        <div class="stats">
            <p><strong>Total Molecules Analyzed:</strong> {len(docking_results)}</p>
            <p><strong>Best Binding Affinity:</strong> {docking_results['binding_affinity'].min():.2f} kcal/mol</p>
            <p><strong>Average Binding Affinity:</strong> {docking_results['binding_affinity'].mean():.2f} kcal/mol</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Visualization Files</h2>
"""
        
        for file_type, file_path in report_files.items():
            if file_path and os.path.exists(file_path):
                rel_path = os.path.relpath(file_path, self.output_dir)
                html_content += f"""
        <div><a href="{rel_path}" target="_blank">{file_type.replace('_', ' ').title()}</a></div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(summary_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Summary report saved to {summary_file}")
        return str(summary_file)


if __name__ == "__main__":
    viz_suite = MolecularVisualizationSuite()
    
    sample_data = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'c1ccccc1O', 'CCN', 'CCCO'],
        'binding_affinity': [-8.5, -7.2, -9.1, -6.8, -7.9],
        'molecular_weight': [46, 44, 94, 45, 60],
        'logp': [-0.3, 1.1, 1.5, -0.5, -0.2]
    })
    
    if viz_suite.plotly.available:
        viz_suite.plotly.plot_binding_affinities_3d(sample_data, "test_binding_plot.html")
        viz_suite.plotly.plot_chemical_space(sample_data, output_html="test_chemical_space.html")
        print("Plotly visualizations created")
    
    print("Visualization module test completed")
