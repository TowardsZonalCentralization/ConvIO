"""
PDF Report Generator for CONVIO
================================
Description:
This module provides the `ReportGenerator` class, responsible for creating a
comprehensive, professional, and standards-compliant technical report in PDF
format. It encapsulates all the logic for document styling, data presentation,
and layout, separating the reporting concerns from the main application's
computational logic.


Key Classes:
------------
- ReportGenerator: A class that takes the application's data and orchestrates
  the creation of the PDF document, section by section.


Usage:
------
An instance of `ReportGenerator` is created with a reference to the main
application object. The public `generate_pdf(file_path)` method is then called
to build and save the entire document.
"""
import os
import io
from datetime import datetime

import networkx as nx
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

class ReportGenerator:
    """
    Handles the creation of a detailed PDF report for wiring harness optimization.

    This class encapsulates all logic related to PDF generation, ensuring a
    separation of concerns from the main application logic. It structures the
    report to align with professional documentation standards, including sections
    like Abstract, Methodology, and Results, with numbered figures and tables.
    """
    def __init__(self, app_instance):
        """
        Initializes the ReportGenerator.

        Args:
            app_instance: A reference to the main WiringHarnessOptimizer instance,
                          providing access to the data and configuration needed
                          for the report.
        """
        self.app = app_instance
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Initialize counters for automatic numbering of sections, figures, and tables.
        # This ensures consistency and adherence to standard report formats.
        self.sec_counter = 1
        self.fig_counter = 1
        self.tbl_counter = 1
        
        # Define custom paragraph styles for a consistent look.
       
        self._setup_styles()

    def _setup_styles(self):
        """Creates and registers custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='Justify', 
            parent=self.styles['BodyText'], 
            alignment=TA_JUSTIFY
        ))
        self.styles.add(ParagraphStyle(
            name='Caption', 
            parent=self.styles['Normal'], 
            alignment=TA_CENTER, 
            spaceBefore=6, 
            fontSize=9
        ))
        self.styles.add(ParagraphStyle(
            name='Abstract', 
            parent=self.styles['Italic'], 
            alignment=TA_JUSTIFY, 
            leftIndent=inch*0.5, 
            rightIndent=inch*0.5
        ))

    def generate_pdf(self, file_path: str):
        """
        Orchestrates the generation of the PDF report and saves it to disk.

        This is the main public method of the class. It builds the report
        sequentially by calling private methods for each section.

        Args:
            file_path (str): The absolute path where the PDF report will be saved.
        """
        # Configure the document template with standard A4 size and 1-inch margins.
        doc = SimpleDocTemplate(file_path, pagesize=A4,
                                leftMargin=inch, rightMargin=inch,
                                topMargin=inch, bottomMargin=inch)
        
        # Build the report section by section. The `self.story` list accumulates
        # all the ReportLab Flowables (paragraphs, tables, images).
        self._add_title_page()
        self._add_abstract()
        self._add_introduction()
        self._add_methodology()
        self._add_algorithmic_framework_section()
        self._add_results()
        self._add_conclusion()
    

        # The build method renders the story into a PDF document.
        doc.build(self.story)

    def _add_title_page(self):
        self.story.append(Paragraph("Detailed Report on the results from CONVIO", self.styles["Title"]))
        self.story.append(Spacer(1, 0.25*inch))
        self.story.append(Paragraph("Technical Report", self.styles["h2"]))
        self.story.append(Spacer(1, 1*inch))
        self.story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", self.styles["Normal"]))
        self.story.append(Paragraph(f"<b>Input Chassis File:</b> {os.path.basename(self.app.chassis_file_path or 'N/A')}", self.styles["Normal"]))
        self.story.append(Paragraph(f"<b>Input I/O File:</b> {os.path.basename(self.app.io_file_path or 'N/A')}", self.styles["Normal"]))
        
        # Determine which linkage method to display
        if self.app.comparison_results:
            linkage_method = self.app.comparison_results.get("best_method", "N/A")
            self.story.append(Paragraph(f"<b>Optimal Linkage Method (Recommended):</b> {linkage_method}", self.styles["Normal"]))
        else:
            linkage_method = self.app.linkage_combo.currentText()
            self.story.append(Paragraph(f"<b>Clustering Linkage Method (Manual Run):</b> {linkage_method}", self.styles["Normal"]))

        self.story.append(PageBreak())

    def _add_abstract(self):
        abstract_text = (
            "<b><i>This report presents a computational analysis for the optimization of automotive "
            "wiring harnesses. A graph-based representation of a vehicle chassis is utilized to compare a "
            "traditional point-to-point wiring architecture against a modern Zonal Electrical/Electronic (EE) "
            "Architecture with I/O aggregators Concept. The methodology employs Agglomerative Clustering on a precomputed graph-distance matrix, "
            "informed by the elbow method, to group Input/Output (I/O) points into optimal zones. Dijkstra's algorithm "
            "is then used to calculate the shortest wiring paths. Quantitative results, including total wire length "
            "and estimated cost, are presented to evaluate the efficacy of the zonal approach. </i></b>"
        )
        self.story.append(Paragraph(abstract_text, self.styles['Abstract']))
        self.story.append(Spacer(1, 0.25*inch))

    def _add_introduction(self):
        self.story.append(Paragraph(f"{self.sec_counter}. INTRODUCTION", self.styles["h1"]))
        self.sec_counter += 1
        intro_text = (
            "The increasing complexity of modern vehicles, driven by advanced driver-assistance systems (ADAS), "
            "infotainment, and vehicle connectivity, has led to a proportional increase in the complexity of "
            "the automotive wiring harness. Traditional wiring architectures, which connect each component directly "
            "to a central computer, are becoming untenable due to weight, cost, and manufacturing challenges. "
            "The Zonal EE Architecture with I/O aggregator concept paradigm addresses these issues by decentralizing intelligence and I/O "
            "management into regional zones. This report documents the process and results of a software tool, "
            "CONVIO, designed to model, analyze, and optimize such architectures."
        )
        self.story.append(Paragraph(intro_text, self.styles['Justify']))
        self.story.append(Spacer(1, 0.25*inch))

    def _add_methodology(self):
        """Adds the Methodology section, detailing the function of each software module."""
        self.story.append(Paragraph(f"{self.sec_counter}. METHODOLOGY", self.styles["h1"]))
        sub_sec = 1
        
        # Subsection 2.1: System Modeling (graph_loader.py)
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} System Modeling (`graph_loader.py`)", self.styles["h2"]))
        sub_sec += 1
        method_text = (
            "The physical structure of the vehicle is first translated into a computational model. "
            "The `graph_loader` module ingests a JSON file describing the vehicle chassis as a series of nodes (structural points) "
            "and edges (valid pathways for wires), forming a weighted, undirected graph. A separate CSV file containing the 2D coordinates "
            "of all I/O devices is then processed. Each I/O point is intelligently mapped onto the chassis graph, either by connecting "
            "it to the nearest existing node or by projecting it onto the closest edge, creating a new node at that point. "
            "This process results in a unified network graph that serves as the basis for all subsequent analysis."
        )
        self.story.append(Paragraph(method_text, self.styles['Justify']))
        self.story.append(Spacer(1, 0.1*inch))

        # Subsection 2.2: Optimal Cluster Analysis (elbow_method.py)
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Optimal Cluster Analysis (`elbow_method.py`)", self.styles["h2"]))
        sub_sec += 1
        method_text = (
            "To determine the most efficient number of I/O aggregators (clusters) for the Zonal EE Architecture, a quantitative approach is employed. "
            "The `elbow_method` module applies the K-Means clustering algorithm specifically to the 2D spatial data of the I/O nodes for a range of cluster counts (k). "
            "It calculates the Within-Cluster Sum of Squares (WCSS) for each k. The 'elbow point'—the point on the WCSS curve where the rate of decrease sharply "
            "changes—is identified as a data-driven suggestion for the optimal number of clusters. This balances harness simplification against the number of required I/O aggregators."
        )
        self.story.append(Paragraph(method_text, self.styles['Justify']))
        self.story.append(Spacer(1, 0.1*inch))

        # Subsection 2.3: Zonal Architecture Optimization (clustering_dijkstra.py)
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Zonal Architecture Optimization (`clustering_dijkstra.py`)", self.styles["h2"]))
        sub_sec += 1
        method_text = (
            "With the optimal number of clusters determined, the `clustering_dijkstra` module executes a sophisticated two-stage optimization. "
            "<b>First</b>, it constructs a distance matrix representing the true shortest-path distance within the chassis graph between every pair of I/O nodes. "
            "Using this matrix, it performs an initial partitioning of the I/O nodes using Agglomerative Clustering to establish a strong baseline grouping. "
            "<b>Second</b>, it enters a refinement phase. For each initial cluster, an optimal I/O aggregator location (centroid) is calculated. The algorithm then iteratively re-assigns each I/O node to the zone with the closest centroid, ensuring that the final groupings are not just based on I/O-to-I/O proximity, but on the more critical I/O-to-aggregator wiring distance. This process repeats until the cluster memberships stabilize, guaranteeing a more logically and efficiently partitioned zonal architecture."
        )
        self.story.append(Paragraph(method_text, self.styles['Justify']))
        self.story.append(Spacer(1, 0.1*inch))

        # Subsection 2.4: Baseline Architecture Analysis (hpc_connector.py)
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Baseline Architecture Analysis (`hpc_connector.py`)", self.styles["h2"]))
        sub_sec += 1
        method_text = (
            "To provide a benchmark for evaluation, the `hpc_connector` module calculates the total wiring length of a traditional, "
            "non-zonal architecture. In this baseline scenario, it is assumed that every I/O node is wired directly to the High-Performance "
            "Computer (HPC). The module computes the shortest path on the graph from each I/O node to the designated HPC node. The sum of the lengths "
            "of these paths represents the total harness length for the baseline architecture, against which the zonal model is compared."
        )
        self.story.append(Paragraph(method_text, self.styles['Justify']))
        
        self.sec_counter += 1
        self.story.append(PageBreak())

    def _add_algorithmic_framework_section(self):
        """Adds a section detailing the key algorithms used in the analysis."""
        self.story.append(Paragraph(f"{self.sec_counter}. ALGORITHMIC FRAMEWORK", self.styles["h1"]))
        self.story.append(Paragraph(
            "The optimization process is underpinned by several key algorithms and data structures, each chosen for a specific purpose. "
            "This section details the role and rationale for each.",
            self.styles['Justify']
        ))
        self.story.append(Spacer(1, 0.2*inch))
        sub_sec = 1

        # --- Algorithm 1: KD-Tree ---
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} KD-Tree for Nearest Neighbor Search", self.styles["h2"]))
        sub_sec += 1
        self.story.append(Paragraph("<b>Algorithm:</b> KD-Tree (k-dimensional tree)", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Purpose:</b> To perform efficient nearest neighbor searches when mapping I/O nodes to the chassis graph.", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Why:</b> A KD-Tree organizes points in a multi-dimensional space to enable exceptionally fast queries (logarithmic time complexity on average). This is critically important for performance, as it avoids a slow, brute-force comparison of every I/O point against every chassis node.", self.styles['Justify']))
        self.story.append(Spacer(1, 0.15*inch))

        # --- Algorithm 2: K-Means Clustering ---
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} K-Means Clustering", self.styles["h2"]))
        sub_sec += 1
        self.story.append(Paragraph("<b>Algorithm:</b> K-Means Clustering", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Purpose:</b> To provide a data-driven suggestion for the optimal number of clusters (k).", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Why:</b> K-Means is a fast heuristic used in the 'elbow method' to analyze the spatial distribution of I/O nodes. By measuring the Within-Cluster Sum of Squares (WCSS) for a range of k values, it helps identify the point of diminishing returns, providing a strong, objective recommendation for the number of zones required.", self.styles['Justify']))
        self.story.append(Spacer(1, 0.15*inch))

        # --- Algorithm 3: Dijkstra's Algorithm ---
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Dijkstra's Shortest Path Algorithm", self.styles["h2"]))
        sub_sec += 1
        self.story.append(Paragraph("<b>Algorithm:</b> Dijkstra's Algorithm", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Purpose:</b> To calculate the shortest path distances between nodes on the chassis graph.", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Why:</b> Wiring cannot pass through empty space; it must follow the vehicle's structure. Dijkstra's algorithm finds the shortest path by traversing the edges of the graph, respecting its constraints. It is essential for calculating the true wiring distances used in the final clustering step and for determining the length of all wiring paths.", self.styles['Justify']))
        self.story.append(Spacer(1, 0.15*inch))

        # --- Algorithm 4: Agglomerative Clustering ---
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Agglomerative Hierarchical Clustering", self.styles["h2"]))
        sub_sec += 1
        self.story.append(Paragraph("<b>Algorithm:</b> Agglomerative Clustering", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Purpose:</b> To perform the final, high-fidelity clustering of I/O nodes into zones.", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Why:</b> Unlike K-Means, Agglomerative Clustering can operate on a precomputed distance matrix. This is its key advantage here. We provide it with a matrix of true wiring distances calculated by Dijkstra's algorithm. This ensures clusters are based on actual path lengths through the chassis, resulting in more practical and accurately optimized zones.", self.styles['Justify']))
        self.story.append(Spacer(1, 0.15*inch))

        # --- Algorithm 5: Greedy Nearest Neighbor ---
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Greedy Nearest Neighbor Heuristic", self.styles["h2"]))
        sub_sec += 1
        self.story.append(Paragraph("<b>Algorithm:</b> Greedy Nearest Neighbor", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Purpose:</b> To calculate a practical and efficient path for the CAN bus connecting all I/O aggregators.", self.styles['BodyText']))
        self.story.append(Paragraph("<b>Why:</b> Finding the absolute shortest path to connect multiple points is a computationally hard problem (Traveling Salesperson Problem). A greedy nearest neighbor heuristic provides an excellent and fast approximation. It constructs the path by iteratively traveling from the current point to the nearest unvisited I/O aggregator, ensuring a short and logical bus topology without excessive computation.", self.styles['Justify']))

        self.sec_counter += 1
        self.story.append(PageBreak())

    def _add_results(self):
        self.story.append(Paragraph(f"{self.sec_counter}. RESULTS", self.styles["h1"]))
        sub_sec = 1
        
        # Graph Analysis
        self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} System Model Graph Analysis", self.styles["h2"]))
        sub_sec += 1
        self._add_graph_table()
        
        # Linkage Method Comparison
        if self.app.comparison_results:
            self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Linkage Method Comparison", self.styles["h2"]))
            sub_sec += 1
            self._add_linkage_comparison_section()

        # Elbow Analysis
        if self.app.elbow_data:
            self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Optimal Cluster Determination", self.styles["h2"]))
            sub_sec += 1
            self._add_elbow_figure()
        
        # Baseline Architecture
        if self.app.hpc_results:
            self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Baseline Architecture Analysis", self.styles["h2"]))
            sub_sec += 1
            self._add_hpc_table()
            self._add_hpc_figure()

        # Optimized Architecture
        if self.app.clustering_results:
            self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Optimized Zonal Architecture with I/O aggregators Analysis", self.styles["h2"]))
            sub_sec += 1
            self._add_clustering_tables()
            # The main figure will be of the best result if available, otherwise the manual run
            self._add_clustering_figure()
        
        # Comparison
        if self.app.hpc_results and self.app.clustering_results:
            self.story.append(Paragraph(f"{self.sec_counter}.{sub_sec} Comparative Analysis", self.styles["h2"]))
            self._add_comparison_table()
        
        self.sec_counter += 1

    def _add_conclusion(self):
        self.story.append(Paragraph(f"{self.sec_counter}. CONCLUSION", self.styles["h1"]))
        self.sec_counter += 1
        conc_text = ""
        if self.app.hpc_results and self.app.clustering_results:
            hpc_length = self.app.hpc_results.get("total_length", 0.0)
            cluster_length = self.app.clustering_results.get("overall_wiring_harness_length", 0.0)
            if hpc_length > 0:
                improvement = ((hpc_length - cluster_length) / hpc_length) * 100
                hpc_weight = (hpc_length / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)
                cluster_weight = (cluster_length / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)
                weight_improvement = ((hpc_weight - cluster_weight) / hpc_weight) * 100 if hpc_weight > 0 else 0
                
                conc_text = (
                    f"The computational analysis confirms the viability and benefits of a Zonal EE Architecture. "
                    f"The optimized model yielded a wiring reduction of {improvement:.1f}% in length and "
                    f"{weight_improvement:.1f}% in weight, translating to significant potential savings in "
                    f"material cost, weight, and manufacturing complexity."
                )
        else:
            conc_text = "The analyses performed provide a foundational dataset for wiring harness optimization. To reach a definitive conclusion, all analytical steps, including baseline and zonal architecture calculations, must be completed to enable a comparative assessment."
        
        self.story.append(Paragraph(conc_text, self.styles['Justify']))
        self.story.append(PageBreak())


    # --- Helper methods for adding specific tables and figures ---

    def _add_graph_table(self):
        stats = self.app._get_graph_statistics(self.app.current_graph)
        stats_data = [
            [Paragraph(c, self.styles["Normal"]) for c in ["Metric", "Value", "Description"]],
            [Paragraph("Total Nodes", self.styles["Normal"]), stats.get("total_nodes", "N/A"), Paragraph("All points in the graph.", self.styles["Normal"])],
            [Paragraph("Total Edges", self.styles["Normal"]), stats.get("total_edges", "N/A"), Paragraph("Connections between chassis nodes.", self.styles["Normal"])],
            [Paragraph("I/O Nodes", self.styles["Normal"]), stats.get("io_nodes", "N/A"), Paragraph("Represents sensors, actuators, etc.", self.styles["Normal"])],
            [Paragraph("Chassis Nodes", self.styles["Normal"]), stats.get("chassis_nodes", "N/A"), Paragraph("Structural points of the vehicle frame.", self.styles["Normal"])],
        ]
        table = Table(stats_data, colWidths=[1.5*inch, 1*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgrey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(Paragraph(f"TABLE {self.tbl_counter}. SYSTEM GRAPH METRICS", self.styles['Caption']))
        self.tbl_counter += 1
        self.story.append(PageBreak())

    def _add_elbow_figure(self):
        optimal_k = self.app.elbow_data.get('elbow_k', 'N/A')
        self.story.append(Paragraph(f"The elbow method analysis (Fig. {self.fig_counter}) was used to determine an optimal number of clusters. The 'elbow point' was identified at k={optimal_k}, indicating the point of diminishing returns for adding more clusters.", self.styles["Justify"]))
        try:
            fig = self.app.elbow_widget.figure
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300)
            img_buffer.seek(0)
            img = Image(img_buffer, width=5*inch, height=3.5*inch)
            self.story.append(img)
            self.story.append(Paragraph(f"Fig. {self.fig_counter}. Elbow Method for Optimal k", self.styles['Caption']))
            self.fig_counter += 1
        except Exception as e:
            self.story.append(Paragraph(f"<i>Could not render elbow plot: {e}</i>", self.styles["Italic"]))
        self.story.append(PageBreak())

    def _add_hpc_figure(self):
        """Adds the HPC wiring visualization to the report."""
        self.story.append(Paragraph(f"The visualization of the baseline architecture is shown in Fig. {self.fig_counter}. Each red line represents the shortest path from an I/O node directly to the central HPC.", self.styles["Justify"]))
        try:
            # Call the helper method on the main app to export the GUI plot
            image_bytes = self.app._export_plot_to_image_bytes(self.app.hpc_view)
            img_buffer = io.BytesIO(image_bytes)
            
            img = Image(img_buffer, width=6*inch, height=4.5*inch, kind='proportional')
            self.story.append(img)
            self.story.append(Paragraph(f"Fig. {self.fig_counter}. Baseline Direct-to-HPC Wiring Architecture", self.styles['Caption']))
            self.fig_counter += 1
        except Exception as e:
            self.story.append(Paragraph(f"<i>Could not render HPC plot: {e}</i>", self.styles["Italic"]))
        self.story.append(PageBreak())

    def _add_clustering_figure(self):
        """Adds the Zonal EEA visualization to the report."""
        title = "Optimized Zonal EEA with I/O aggregators"
        if self.app.comparison_results:
            best_method = self.app.comparison_results.get("best_method", "N/A")
            title = f"Optimized Zonal EEA (Recommended Method: '{best_method}')"

        self.story.append(Paragraph(f"The visualization of the optimized Zonal EE Architecture is shown in Fig. {self.fig_counter}. I/O nodes are color-coded by cluster, with wiring paths shown to their respective I/O aggregators (centroids).", self.styles["Justify"]))
        try:
            image_bytes = self.app._export_plot_to_image_bytes(self.app.cluster_view)
            img_buffer = io.BytesIO(image_bytes)

            img = Image(img_buffer, width=6*inch, height=4.5*inch, kind='proportional')
            self.story.append(img)
            self.story.append(Paragraph(f"Fig. {self.fig_counter}. {title}", self.styles['Caption']))
            self.fig_counter += 1
        except Exception as e:
            self.story.append(Paragraph(f"<i>Could not render clustering plot: {e}</i>", self.styles["Italic"]))
        self.story.append(PageBreak())

    def _add_hpc_table(self):
        total_length = self.app.hpc_results.get("total_length", 0)
        cost = (total_length / 1000.0) * self.app.cost_cfg.get("wire_price_per_m", 0.0)
        weight = (total_length / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)
        data = [
            ["Metric", "Value"],
            ["Total Wiring Length", f"{total_length:.2f} mm ({total_length/1000:.2f} m)"],
            ["Estimated Wiring Cost", f"{cost:.2f} {self.app.cost_cfg.get('currency', '')}"],
            ["Estimated Wiring Weight", f"{weight:.3f} kg"],
        ]
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgrey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(Paragraph(f"TABLE {self.tbl_counter}. BASELINE ARCHITECTURE METRICS", self.styles['Caption']))
        self.tbl_counter += 1
        self.story.append(PageBreak())

    def _add_clustering_tables(self):
        # If a full comparison was run, show the detailed tables for all methods.
        if self.app.comparison_results:
            results_to_process = self.app.comparison_results.get("results", {})
            best_method = self.app.comparison_results.get("best_method")
            
            self.story.append(Paragraph("Detailed Breakdown by Linkage Method", self.styles["h3"]))
            self.story.append(Spacer(1, 0.1*inch))

            # Sort to ensure a consistent order in the report
            for method in sorted(results_to_process.keys()):
                results = results_to_process[method]
                if not results: continue
                
                is_best = " (Recommended)" if method == best_method else ""
                self.story.append(Paragraph(f"<b>Results for '{method}' Linkage{is_best}</b>", self.styles["h3"]))
                
                self._generate_tables_for_single_result(results, method_name=method)
                self.story.append(PageBreak())

        # Fallback for a single, manual run
        elif self.app.clustering_results:
            self._generate_tables_for_single_result(self.app.clustering_results)

    def _generate_tables_for_single_result(self, clustering_results, method_name: str = ""):
        """
        Helper to generate the set of tables and the corresponding plot for a
        single clustering result dict.
        """
        wire_len = clustering_results.get("total_wire_length", 0)
        can_len = clustering_results.get("can_bus", {}).get("total_length", 0)
        total_len = clustering_results.get("overall_wiring_harness_length", 0)
        wire_cost = (wire_len / 1000.0) * self.app.cost_cfg.get("wire_price_per_m", 0.0)
        can_cost = (can_len / 1000.0) * self.app.cost_cfg.get("CAN_bus_price_per_m", 0.0)
        wire_weight = (wire_len / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)
        can_weight = (can_len / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0) # Assuming CAN bus has same weight per meter
        currency = self.app.cost_cfg.get("currency", "")
        summary_data = [
            ["Component", "Length (mm)", f"Cost ({currency})", "Weight (kg)"],
            ["I/O Wiring", f"{wire_len:.2f}", f"{wire_cost:.2f}", f"{wire_weight:.3f}"],
            ["CAN Bus", f"{can_len:.2f}", f"{can_cost:.2f}", f"{can_weight:.3f}"],
            ["Total", f"{total_len:.2f}", f"{wire_cost + can_cost:.2f}", f"{wire_weight + can_weight:.3f}"],
        ]
        table = Table(summary_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgrey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(Paragraph(f"TABLE {self.tbl_counter}. ZONAL ARCHITECTURE COST, LENGTH & WEIGHT SUMMARY", self.styles['Caption']))
        self.tbl_counter += 1
        self.story.append(Spacer(1, 0.2*inch))

        clusters = clustering_results.get("clusters", {})
        path_style = self.styles["Code"].clone('PathStyle', fontSize=7, leading=8)
        
        # Check if there are any clusters to detail
        if not any(cid != "can_bus" for cid in clusters):
            self.story.append(Paragraph("No detailed cluster data to display for this method.", self.styles['Normal']))
            return

        for cid, cdata in sorted(clusters.items()):
            if cid == "can_bus": continue
            cluster_id_str = cid.split('_')[-1]
            self.story.append(Paragraph(f"<b>Cluster {cluster_id_str} Wiring Details</b>", self.styles["h4"]))
            wiring_paths = cdata.get("wiring_paths", {})
            path_data = [[Paragraph(c, self.styles["Normal"]) for c in ["I/O Node", "Length (mm)", "Shortest Path"]]]
            for io_node in sorted(wiring_paths.keys()):
                path_info = wiring_paths[io_node]
                path_data.append([
                    Paragraph(io_node, self.styles["Normal"]),
                    f"{path_info.get('length', 0):.2f}",
                    Paragraph(' -> '.join(path_info.get("path", [])), path_style)
                ])
            path_table = Table(path_data, colWidths=[1*inch, 1*inch, 4*inch], repeatRows=1)
            path_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black), ('BOX', (0,0), (-1,-1), 0.25, colors.black),
            ]))
            self.story.append(path_table)
            self.story.append(Paragraph(f"TABLE {self.tbl_counter}. WIRING PATHS FOR CLUSTER {cluster_id_str}", self.styles['Caption']))
            self.tbl_counter += 1
            self.story.append(Spacer(1, 0.2*inch))
        
        # Add the plot for this specific method if it's part of a comparison
        if self.app.comparison_results and method_name:
            self.story.append(Spacer(1, 0.2*inch))
            try:
                title = f"Zonal EEA Visualization for '{method_name}' Linkage"
                image_bytes = self.app.generate_clustering_plot_for_report(clustering_results, title)
                img_buffer = io.BytesIO(image_bytes)
                img = Image(img_buffer, width=6*inch, height=4.5*inch, kind='proportional')
                self.story.append(img)
                self.story.append(Paragraph(f"Fig. {self.fig_counter}. {title}", self.styles['Caption']))
                self.fig_counter += 1
            except Exception as e:
                self.story.append(Paragraph(f"<i>Could not render plot for '{method_name}': {e}</i>", self.styles["Italic"]))

    def _add_linkage_comparison_section(self):
        """Adds the linkage method comparison table and text to the report."""
        self.story.append(Paragraph(
            "To ensure the most optimal clustering strategy, a comprehensive analysis was performed, comparing three standard linkage methods. "
            "The table below summarizes the final wiring harness length for each method after the full optimization and refinement process. "
            "The method yielding the shortest overall length is recommended and used for all subsequent detailed analysis in this report.",
            self.styles['Justify']
        ))
        self.story.append(Spacer(1, 0.15*inch))

        results = self.app.comparison_results.get("results", {})
        best_method = self.app.comparison_results.get("best_method")
        
        data = [
            [Paragraph(c, self.styles["Normal"]) for c in ["Linkage Method", "I/O Wiring (mm)", "CAN Bus (mm)", "Total Length (mm)"]]
        ]
        
        methods = ["average", "complete", "single"]
        for method in methods:
            res = results.get(method, {})
            wire_len = res.get("total_wire_length", 0.0)
            can_len = res.get("can_bus", {}).get("total_length", 0.0)
            total_len = res.get("overall_wiring_harness_length", 0.0)
            
            row_data = [
                Paragraph(f"<b>{method}</b>" if method == best_method else method, self.styles["Normal"]),
                Paragraph(f"<b>{wire_len:.2f}</b>" if method == best_method else f"{wire_len:.2f}", self.styles["Normal"]),
                Paragraph(f"<b>{can_len:.2f}</b>" if method == best_method else f"{can_len:.2f}", self.styles["Normal"]),
                Paragraph(f"<b>{total_len:.2f}</b>" if method == best_method else f"{total_len:.2f}", self.styles["Normal"]),
            ]
            data.append(row_data)

        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgrey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0, methods.index(best_method)+1), (-1, methods.index(best_method)+1), colors.HexColor('#d4edda'))
        ]))
        self.story.append(table)
        self.story.append(Paragraph(f"TABLE {self.tbl_counter}. LINKAGE METHOD COMPARISON", self.styles['Caption']))
        self.tbl_counter += 1
        self.story.append(PageBreak())

    def _add_comparison_table(self):
        hpc_len = self.app.hpc_results.get("total_length", 0.0)
        hpc_cost = (hpc_len / 1000.0) * self.app.cost_cfg.get("wire_price_per_m", 0.0)
        hpc_weight = (hpc_len / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)
        
        zonal_len = self.app.clustering_results.get("overall_wiring_harness_length", 0.0)
        zonal_cost = ((self.app.clustering_results.get("total_wire_length", 0) / 1000.0) * self.app.cost_cfg.get("wire_price_per_m", 0.0)) + \
                     ((self.app.clustering_results.get("can_bus", {}).get("total_length", 0) / 1000.0) * self.app.cost_cfg.get("CAN_bus_price_per_m", 0.0))
        zonal_weight = ((self.app.clustering_results.get("total_wire_length", 0) / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0)) + \
                       ((self.app.clustering_results.get("can_bus", {}).get("total_length", 0) / 1000.0) * self.app.cost_cfg.get("wire_weight_per_m_kg", 0.0))
        
        len_reduc = hpc_len - zonal_len
        cost_reduc = hpc_cost - zonal_cost
        weight_reduc = hpc_weight - zonal_weight
        
        currency = self.app.cost_cfg.get("currency", "")
        data = [
            ["Metric", "Direct HPC Wiring", "Zonal EEA", "Improvement"],
            ["Total Length (mm)", f"{hpc_len:.2f}", f"{zonal_len:.2f}", f"{len_reduc:.2f} ({ (len_reduc/hpc_len)*100 if hpc_len>0 else 0 :.1f}%)"],
            [f"Total Cost ({currency})", f"{hpc_cost:.2f}", f"{zonal_cost:.2f}", f"{cost_reduc:.2f} ({ (cost_reduc/hpc_cost)*100 if hpc_cost>0 else 0 :.1f}%)"],
            ["Total Weight (kg)", f"{hpc_weight:.3f}", f"{zonal_weight:.3f}", f"{weight_reduc:.3f} ({ (weight_reduc/hpc_weight)*100 if hpc_weight>0 else 0 :.1f}%)"],
        ]
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.black), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(Paragraph(f"TABLE {self.tbl_counter}. COMPARATIVE ANALYSIS OF ARCHITECTURES", self.styles['Caption']))
        self.tbl_counter += 1
        self.story.append(PageBreak())
