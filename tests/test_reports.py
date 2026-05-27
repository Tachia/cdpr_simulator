"""Report generation: captioned figures, tables, markdown summary."""

from __future__ import annotations

from cdpr.recording import load_experiment, record_simulation
from cdpr.reports import (
    CaptionedFigure,
    cable_summary_table,
    save_captioned_figure,
    summary_table_csv,
    summary_table_latex,
    write_markdown_summary,
)
from cdpr.viz import plots2d


def _make_experiment(robot, sim, tmp_path):
    log = record_simulation(robot=robot, result=sim, out_dir=tmp_path / "exp",
                            title="reports test")
    return load_experiment(log.root)


def test_save_captioned_figure_writes_pdf_png_tex(short_sim, ipanema, tmp_path):
    fig = plots2d.plot_cable_tensions(short_sim, robot=ipanema)
    cf = CaptionedFigure(
        figure=fig,
        caption=r"Cable tensions $T_i(t)$ during the smoke-test hold.",
        label="fig:tensions",
    )
    written = save_captioned_figure(cf, tmp_path)
    assert "pdf" in written and written["pdf"].exists()
    assert "png" in written and written["png"].exists()
    assert "tex" in written and r"\includegraphics" in written["tex"].read_text()


def test_summary_table_csv_and_latex(short_sim, ipanema, tmp_path):
    exp = _make_experiment(ipanema, short_sim, tmp_path)

    csv_path = summary_table_csv(exp, tmp_path / "summary.csv")
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "channel,mean,std,min,max" in text

    tex_path = summary_table_latex(exp, tmp_path / "summary.tex",
                                   caption="Channel statistics.")
    assert tex_path.exists()
    assert r"\begin{table}" in tex_path.read_text()


def test_cable_summary_table_two_formats(short_sim, ipanema, tmp_path):
    exp = _make_experiment(ipanema, short_sim, tmp_path)
    csv_path = cable_summary_table(exp, tmp_path / "cables.csv", fmt="csv")
    assert csv_path.exists()
    tex_path = cable_summary_table(exp, tmp_path / "cables.tex", fmt="latex")
    assert r"\begin{tabular}" in tex_path.read_text()


def test_markdown_summary_contains_expected_sections(short_sim, ipanema, tmp_path):
    exp = _make_experiment(ipanema, short_sim, tmp_path)
    md_path = write_markdown_summary(exp, tmp_path / "summary.md",
                                     figures=[("Cable tensions", "tensions.png")])
    txt = md_path.read_text()
    for header in ("## Experiment", "## Robot", "## Simulation",
                   "## Reproducibility", "## Results", "## Figures"):
        assert header in txt
