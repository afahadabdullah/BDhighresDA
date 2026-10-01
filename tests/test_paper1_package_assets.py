"""Updated manuscript figures must not be selected by an old figure manifest."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("paper_package",Path(__file__).resolve().parents[1]/"scripts/93_package_cpcv2_paper1.py")
mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class PackageAssetTests(unittest.TestCase):
    def test_current_graphics_optional_slots_and_comments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root/"figures").mkdir();(root/"additional").mkdir()
            (root/"figures/fig01_overview.pdf").write_bytes(b"actual-new-figure")
            (root/"additional/tab_product_daily.tex").write_text("actual derived table")
            source = root/"paper.tex"
            source.write_text(r"""% \includegraphics{obsolete_missing.pdf}
\includegraphics[width=\textwidth]{fig01_overview.pdf}
\IfFileExists{figures/fig08_product_intensity.pdf}{\includegraphics{fig08_product_intensity.pdf}}{pending}
\newcommand{\optionalfigure}[3]{\IfFileExists{additional/#1.pdf}{\includegraphics{additional/#1.pdf}}{pending}}
\optionaltable{tab_product_daily}{title}{instructions}
\optionalfigure{fig_calibration}{title}{instructions}
""")
            files,pending = mod.manuscript_assets(source)
            self.assertEqual({p.name for p in files},{"fig01_overview.pdf","tab_product_daily.tex"})
            self.assertEqual(set(pending),{"fig08_product_intensity.pdf","additional/fig_calibration.pdf"})

    def test_missing_unconditional_figure_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"paper.tex";source.write_text(r"\includegraphics{fig01_overview.pdf}")
            with self.assertRaisesRegex(FileNotFoundError,"required manuscript figure"):
                mod.manuscript_assets(source)


if __name__ == "__main__":unittest.main()
