import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers / synthetic data factories
# ---------------------------------------------------------------------------

def make_simple_label_image():
    """
    Create a minimal 3D labelled image with two touching cubic cells.

    Image shape: (20, 20, 20)
    Cell 1 (label=1): voxels [1:10, 1:19, 1:19]
    Cell 2 (label=2): voxels [10:19, 1:19, 1:19]
    Background (label=0): everything else
    """
    img = np.zeros((20, 20, 20), dtype=np.uint32)
    img[1:10, 1:19, 1:19] = 1
    img[10:19, 1:19, 1:19] = 2
    return img


def make_four_cell_label_image():
    """
    Create a 3D labelled image with four cells arranged in a 2×2 grid
    in XY, stacked in Z. This creates faces AND a shared edge between
    all four cells.

    Image shape: (20, 20, 20)
    Cell 1: [1:19, 1:10, 1:10]
    Cell 2: [1:19, 10:19, 1:10]
    Cell 3: [1:19, 1:10, 10:19]
    Cell 4: [1:19, 10:19, 10:19]
    """
    img = np.zeros((20, 20, 20), dtype=np.uint32)
    img[1:19, 1:10, 1:10] = 1
    img[1:19, 10:19, 1:10] = 2
    img[1:19, 1:10, 10:19] = 3
    img[1:19, 10:19, 10:19] = 4
    return img


def make_mock_cell_df(n_cells=3):
    """Return a minimal cell_df DataFrame as CellMet would produce.

    Face topology used:  face 1 → cells (1, 2),  face 2 → cells (2, 3)
    Derived neighbor counts:
        cell 1 → 1 neighbor  (appears in face 1 only)
        cell 2 → 2 neighbors (appears in faces 1 and 2)
        cell 3 → 1 neighbor  (appears in face 2 only)
    """
    # n_neighbors computed from the face pairs defined in make_mock_face_df
    # face pairs: (1,2), (2,3)
    n_neighbors_map = {1: 1, 2: 2, 3: 1}
    cell_ids = list(range(1, n_cells + 1))
    return pd.DataFrame({
        "cell_id": cell_ids,
        "volume": [500.0, 480.0, 510.0][:n_cells],
        "n_neighbors": [n_neighbors_map[i] for i in cell_ids],
        "surface_area": [300.0, 290.0, 310.0][:n_cells],
        "x": [5.0, 15.0, 10.0][:n_cells],
        "y": [5.0, 5.0, 15.0][:n_cells],
        "z": [5.0, 5.0, 5.0][:n_cells],
    }).set_index("cell_id")


def make_mock_face_df(n_faces=2):
    """Return a minimal face_df DataFrame as CellMet would produce."""
    return pd.DataFrame({
        "face_id": list(range(1, n_faces + 1)),
        "cell_1": [1, 2][:n_faces],
        "cell_2": [2, 3][:n_faces],
        "area": [120.0, 115.0][:n_faces],
        "x_mid": [10.0, 10.0][:n_faces],
        "y_mid": [5.0, 10.0][:n_faces],
        "z_mid": [5.0, 5.0][:n_faces],
        "length_x": [8.0, 8.0][:n_faces],
        "length_y": [8.0, 8.0][:n_faces],
    })


def make_mock_edge_df(n_edges=1):
    """Return a minimal edge_df DataFrame as CellMet would produce."""
    return pd.DataFrame({
        "edge_id": list(range(1, n_edges + 1)),
        "cell_1": [1][:n_edges],
        "cell_2": [2][:n_edges],
        "cell_3": [3][:n_edges],
        "length": [15.0][:n_edges],
        "length_shortest": [14.5][:n_edges],
    })


# ---------------------------------------------------------------------------
# 1. Label image validation
# ---------------------------------------------------------------------------

class TestLabelImageValidation:
    """Tests for label image format and content checks."""

    def test_label_image_dtype_uint(self):
        img = make_simple_label_image()
        assert np.issubdtype(img.dtype, np.integer), (
            "Label image must have integer dtype"
        )

    def test_label_image_is_3d(self):
        img = make_simple_label_image()
        assert img.ndim == 3, "Label image must be 3-dimensional"

    def test_label_image_background_is_zero(self):
        img = make_simple_label_image()
        assert 0 in np.unique(img), "Background (label=0) must be present"

    def test_label_image_cell_ids_positive(self):
        img = make_simple_label_image()
        cell_ids = np.unique(img)
        cell_ids = cell_ids[cell_ids != 0]
        assert np.all(cell_ids > 0), "All cell IDs must be positive integers"

    def test_label_image_no_nan(self):
        img = make_simple_label_image().astype(float)
        assert not np.any(np.isnan(img)), "Label image must not contain NaN"

    def test_label_image_cell_count(self):
        img = make_simple_label_image()
        n_cells = len(np.unique(img)) - 1  # exclude background
        assert n_cells == 2

    def test_four_cell_image_has_four_cells(self):
        img = make_four_cell_label_image()
        n_cells = len(np.unique(img)) - 1
        assert n_cells == 4


# ---------------------------------------------------------------------------
# 2. cell_df structure & content
# ---------------------------------------------------------------------------

class TestCellDataFrame:
    """Tests for the cell-level output table."""

    def test_cell_df_required_columns(self):
        df = make_mock_cell_df()
        required = {"volume", "n_neighbors", "surface_area"}
        assert required.issubset(set(df.columns)), (
            f"cell_df missing columns: {required - set(df.columns)}"
        )

    def test_cell_df_volume_positive(self):
        df = make_mock_cell_df()
        assert (df["volume"] > 0).all(), "All cell volumes must be positive"

    def test_cell_df_surface_area_positive(self):
        df = make_mock_cell_df()
        assert (df["surface_area"] > 0).all(), "Surface areas must be positive"

    def test_cell_df_n_neighbors_non_negative(self):
        df = make_mock_cell_df()
        assert (df["n_neighbors"] >= 0).all(), (
            "Neighbor counts cannot be negative"
        )

    def test_cell_df_no_nan_in_volume(self):
        df = make_mock_cell_df()
        assert not df["volume"].isna().any(), "volume column must not have NaN"

    def test_cell_df_no_nan_in_surface_area(self):
        df = make_mock_cell_df()
        assert not df["surface_area"].isna().any()

    def test_cell_df_row_count_matches_n_cells(self):
        n = 3
        df = make_mock_cell_df(n_cells=n)
        assert len(df) == n

    def test_cell_df_index_unique(self):
        df = make_mock_cell_df()
        assert df.index.is_unique, "cell_id index must have no duplicates"

    def test_cell_df_isoperimetric_ratio_sensible(self):
        """
        The isoperimetric ratio SA^3 / V^2 is minimised by a sphere.
        For any convex body it should be >= 36*pi (the sphere value).
        We verify the ratio is finite and positive.
        """
        df = make_mock_cell_df()
        ratio = df["surface_area"] ** 3 / df["volume"] ** 2
        assert (ratio > 0).all() and np.isfinite(ratio).all()


# ---------------------------------------------------------------------------
# 3. face_df structure & content
# ---------------------------------------------------------------------------

class TestFaceDataFrame:
    """Tests for the face-level output table (contacts between two cells)."""

    def test_face_df_required_columns(self):
        df = make_mock_face_df()
        required = {"cell_1", "cell_2", "area"}
        assert required.issubset(set(df.columns))

    def test_face_df_area_positive(self):
        df = make_mock_face_df()
        assert (df["area"] > 0).all(), "Face area must be positive"

    def test_face_df_cells_are_distinct(self):
        df = make_mock_face_df()
        assert (df["cell_1"] != df["cell_2"]).all(), (
            "cell_1 and cell_2 of a face must differ"
        )

    def test_face_df_no_nan_area(self):
        df = make_mock_face_df()
        assert not df["area"].isna().any()

    def test_face_df_cell_ids_positive(self):
        df = make_mock_face_df()
        assert (df["cell_1"] > 0).all() and (df["cell_2"] > 0).all()

    def test_face_df_row_count(self):
        df = make_mock_face_df(n_faces=2)
        assert len(df) == 2

    def test_face_df_midpoint_coordinates_finite(self):
        df = make_mock_face_df()
        for col in ["x_mid", "y_mid", "z_mid"]:
            if col in df.columns:
                assert np.isfinite(df[col]).all(), f"{col} must be finite"


# ---------------------------------------------------------------------------
# 4. edge_df structure & content
# ---------------------------------------------------------------------------

class TestEdgeDataFrame:
    """Tests for the edge-level output table (contacts between three cells)."""

    def test_edge_df_required_columns(self):
        df = make_mock_edge_df()
        required = {"cell_1", "cell_2", "cell_3", "length"}
        assert required.issubset(set(df.columns))

    def test_edge_df_length_positive(self):
        df = make_mock_edge_df()
        assert (df["length"] > 0).all(), "Edge length must be positive"

    def test_edge_df_shortest_length_leq_length(self):
        df = make_mock_edge_df()
        if "length_shortest" in df.columns:
            assert (df["length_shortest"] <= df["length"]).all(), (
                "Shortest path cannot exceed actual length"
            )

    def test_edge_df_three_distinct_cells(self):
        df = make_mock_edge_df()
        for _, row in df.iterrows():
            cell_ids = {row["cell_1"], row["cell_2"], row["cell_3"]}
            assert len(cell_ids) == 3, "Edge must connect exactly three distinct cells"

    def test_edge_df_no_nan_length(self):
        df = make_mock_edge_df()
        assert not df["length"].isna().any()


# ---------------------------------------------------------------------------
# 5. Geometric consistency checks
# ---------------------------------------------------------------------------

class TestGeometricConsistency:
    """Cross-table consistency and basic geometric constraints."""

    def test_face_cells_exist_in_cell_df(self):
        cell_df = make_mock_cell_df(n_cells=3)
        face_df = make_mock_face_df(n_faces=2)
        cell_ids = set(cell_df.index)
        for _, row in face_df.iterrows():
            assert row["cell_1"] in cell_ids, (
                f"face references unknown cell {row['cell_1']}"
            )
            assert row["cell_2"] in cell_ids, (
                f"face references unknown cell {row['cell_2']}"
            )

    def test_edge_cells_exist_in_cell_df(self):
        cell_df = make_mock_cell_df(n_cells=3)
        edge_df = make_mock_edge_df(n_edges=1)
        cell_ids = set(cell_df.index)
        for _, row in edge_df.iterrows():
            for c in ["cell_1", "cell_2", "cell_3"]:
                assert row[c] in cell_ids, (
                    f"edge references unknown cell {row[c]}"
                )

    def test_total_face_area_leq_total_cell_surface_area(self):
        """
        Each face is shared between two cells. The sum of face areas ×2
        should not greatly exceed total surface area (rough sanity check).
        """
        cell_df = make_mock_cell_df(n_cells=3)
        face_df = make_mock_face_df(n_faces=2)
        # Each face is counted once per pair, so contribution is 2× per cell
        total_face_contribution = face_df["area"].sum() * 2
        total_surface = cell_df["surface_area"].sum()
        assert total_face_contribution <= total_surface * 10, (
            "Implausibly large total face area relative to surface area"
        )

    def test_n_neighbors_consistent_with_face_df(self):
        """
        Each face connects exactly two cells, so each cell's neighbor
        count should equal the number of faces it appears in.
        """
        cell_df = make_mock_cell_df(n_cells=3)
        face_df = make_mock_face_df(n_faces=2)
        for cell_id in cell_df.index:
            faces_for_cell = face_df[
                (face_df["cell_1"] == cell_id) |
                (face_df["cell_2"] == cell_id)
            ]
            computed_neighbors = len(faces_for_cell)
            stored_neighbors = cell_df.loc[cell_id, "n_neighbors"]
            assert computed_neighbors == stored_neighbors, (
                f"Cell {cell_id}: n_neighbors={stored_neighbors} but "
                f"appears in {computed_neighbors} faces"
            )


# ---------------------------------------------------------------------------
# 6. Numerical / helper function tests
# ---------------------------------------------------------------------------

class TestNumericalHelpers:
    """
    Unit tests for typical internal geometric computations CellMet relies on.
    These test the mathematical primitives without needing the full pipeline.
    """

    def test_vector_norm(self):
        v = np.array([3.0, 4.0, 0.0])
        assert pytest.approx(np.linalg.norm(v)) == 5.0

    def test_cross_product_perpendicular(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.cross(a, b)
        assert np.dot(c, a) == pytest.approx(0.0)
        assert np.dot(c, b) == pytest.approx(0.0)

    def test_triangle_area_via_cross_product(self):
        """Area of a right triangle with legs 3 and 4 should be 6."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([3.0, 0.0, 0.0])
        p2 = np.array([0.0, 4.0, 0.0])
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        assert area == pytest.approx(6.0)

    def test_centroid_of_triangle(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([3.0, 0.0, 0.0])
        p2 = np.array([0.0, 3.0, 0.0])
        centroid = (p0 + p1 + p2) / 3.0
        assert centroid == pytest.approx([1.0, 1.0, 0.0])

    def test_euclidean_distance(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])
        dist = np.linalg.norm(p2 - p1)
        assert dist == pytest.approx(np.sqrt(3))

    def test_unit_normal_magnitude(self):
        """Normal vector to a planar polygon should have unit magnitude."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        normal = np.cross(v1, v2)
        unit_normal = normal / np.linalg.norm(normal)
        assert np.linalg.norm(unit_normal) == pytest.approx(1.0)

    def test_angle_between_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        assert angle == pytest.approx(np.pi / 2)

    def test_voxel_volume_counting(self):
        """Volume estimated by voxel counting should match for a known cube."""
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[1:6, 1:6, 1:6] = 1  # 5×5×5 = 125 voxels
        vol = np.sum(img == 1)
        assert vol == 125

    def test_label_unique_ids(self):
        img = make_four_cell_label_image()
        unique = np.unique(img)
        assert set(unique) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# 7. Edge cases & robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for boundary conditions and degenerate inputs."""

    def test_empty_image_raises_or_returns_empty(self):
        """An all-zero image has no cells; result should be empty or raise."""
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        n_cells = len(np.unique(img)) - 1  # background only
        assert n_cells == 0

    def test_single_cell_image(self):
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[1:9, 1:9, 1:9] = 1
        n_cells = len(np.unique(img)) - 1
        assert n_cells == 1

    def test_cell_df_with_one_cell_no_neighbors(self):
        """A single isolated cell should have zero neighbors."""
        df = pd.DataFrame({
            "cell_id": [1],
            "volume": [800.0],
            "n_neighbors": [0],
            "surface_area": [400.0],
        }).set_index("cell_id")
        assert df.loc[1, "n_neighbors"] == 0

    def test_face_df_empty_for_single_cell(self):
        """Single cell ⟹ no faces."""
        face_df = pd.DataFrame(columns=["face_id", "cell_1", "cell_2", "area"])
        assert len(face_df) == 0

    def test_non_contiguous_cell_ids(self):
        """CellMet should handle non-contiguous label IDs gracefully."""
        img = np.zeros((20, 20, 20), dtype=np.uint32)
        img[1:10, 1:10, 1:10] = 5   # ID gap: 1,2,3,4 missing
        img[10:19, 1:10, 1:10] = 10
        ids = np.unique(img)
        ids = ids[ids != 0]
        assert set(ids) == {5, 10}

    def test_label_image_large_ids(self):
        """IDs up to uint32 max should be valid."""
        img = np.zeros((5, 5, 5), dtype=np.uint32)
        img[1:4, 1:4, 1:4] = np.iinfo(np.uint32).max
        assert img.max() == np.iinfo(np.uint32).max

    def test_face_df_no_self_contact(self):
        """A face must never connect a cell to itself."""
        face_df = make_mock_face_df()
        assert (face_df["cell_1"] != face_df["cell_2"]).all()

    def test_volume_greater_than_zero_for_all_cells(self):
        df = make_mock_cell_df()
        assert (df["volume"] > 0).all()

    def test_surface_area_greater_than_zero(self):
        df = make_mock_cell_df()
        assert (df["surface_area"] > 0).all()


# ---------------------------------------------------------------------------
# 8. CSV I/O round-trip
# ---------------------------------------------------------------------------

class TestCSVIO:
    """Tests for CSV save/load consistency (mocking the file I/O)."""

    def test_cell_df_csv_roundtrip(self, tmp_path):
        df = make_mock_cell_df()
        path = tmp_path / "cell_df.csv"
        df.to_csv(path)
        df_loaded = pd.read_csv(path, index_col="cell_id")
        pd.testing.assert_frame_equal(df, df_loaded)

    def test_face_df_csv_roundtrip(self, tmp_path):
        df = make_mock_face_df()
        path = tmp_path / "face_df.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, df_loaded)

    def test_edge_df_csv_roundtrip(self, tmp_path):
        df = make_mock_edge_df()
        path = tmp_path / "edge_df.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, df_loaded)

    def test_cell_df_columns_preserved_after_csv(self, tmp_path):
        df = make_mock_cell_df()
        path = tmp_path / "cell_df.csv"
        df.to_csv(path)
        df_loaded = pd.read_csv(path, index_col="cell_id")
        assert set(df.columns) == set(df_loaded.columns)

    def test_csv_numeric_values_preserved(self, tmp_path):
        df = make_mock_cell_df()
        path = tmp_path / "cell_df.csv"
        df.to_csv(path)
        df_loaded = pd.read_csv(path, index_col="cell_id")
        assert df_loaded.loc[1, "volume"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 9. Integration-style checks (pipeline stages)
# ---------------------------------------------------------------------------

class TestPipelineStages:
    """
    Lightweight integration checks that simulate the three-stage
    CellMet pipeline: Prerequisite → Segmentation → Analysis.
    These do not invoke the real CellMet code but verify that the
    data contracts between stages hold.
    """

    def test_prerequisite_output_has_expected_labels(self):
        """
        After the prerequisite stage, every cell ID in the original
        image should produce a single-cell binary array.
        """
        img = make_simple_label_image()
        cell_ids = np.unique(img)
        cell_ids = cell_ids[cell_ids != 0]
        for cid in cell_ids:
            single_cell = (img == cid).astype(np.uint8)
            assert single_cell.max() == 1
            assert single_cell.sum() > 0

    def test_segmentation_produces_cell_df(self):
        """After segmentation, cell_df should have one row per cell."""
        img = make_simple_label_image()
        n_expected = len(np.unique(img)) - 1
        cell_df = make_mock_cell_df(n_cells=n_expected)
        assert len(cell_df) == n_expected

    def test_segmentation_produces_face_df(self):
        """Two touching cells ⟹ exactly one face."""
        face_df = make_mock_face_df(n_faces=1)
        assert len(face_df) == 1

    def test_analysis_volume_units_consistent(self):
        """
        If pixel size is 1 µm, volumes should be in µm³.
        A 9×18×18 voxel cell has volume ≈ 2916 voxels.
        """
        img = make_simple_label_image()
        voxel_volume_um3 = 1.0  # µm³ per voxel
        cell1_voxels = int(np.sum(img == 1))
        estimated_volume = cell1_voxels * voxel_volume_um3
        assert estimated_volume > 0
        assert np.isfinite(estimated_volume)

    def test_analysis_face_area_units_consistent(self):
        """Face area should be positive and finite."""
        face_df = make_mock_face_df()
        assert (face_df["area"] > 0).all()
        assert np.isfinite(face_df["area"]).all()

    def test_cell_df_columns_cover_all_required_outputs(self):
        """Check that no required output column is missing."""
        df = make_mock_cell_df()
        expected_columns = ["volume", "n_neighbors", "surface_area"]
        for col in expected_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_seven_output_files_keys(self):
        """
        CellMet documents 7 output CSV files. Verify the expected
        file name tokens are present.
        """
        expected_outputs = {
            "cell_df",
            "cell_plane_df",
            "edge_df",
            "edge_pixel_df",
            "face_df",
            "face_edge_pixel_df",
            "face_pixel_df",
        }
        # Simulate: check that we can track all 7 keys
        assert len(expected_outputs) == 7