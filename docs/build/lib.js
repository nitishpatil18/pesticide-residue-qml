// Shared helpers for the document builders (docx). Run from the project root:
//   node docs/build/build_report.js
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const D = require("docx");
const R = JSON.parse(fs.readFileSync(path.join(__dirname, "results_summary.json"), "utf8"));
const ROOT = path.resolve(__dirname, "..", "..");
const FONT = "Arial";
const W = 9026; // A4 with 1 inch margins, DXA

const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const pm = (m, s) => `${f3(m)} ± ${f3(s)}`;

const run = (t, o = {}) => new D.TextRun({ text: t, font: FONT, ...o });
function P(text, o = {}) {
  const parts = Array.isArray(text) ? text : [text];
  return new D.Paragraph({
    spacing: { after: 120, line: 300 }, ...o.para,
    children: parts.map((p) => (typeof p === "string" ? run(p, o.run) : p)),
  });
}
const B = (t) => run(t, { bold: true });
const I = (t) => run(t, { italics: true });
const H1 = (t) => new D.Paragraph({ heading: D.HeadingLevel.HEADING_1, children: [run(t)] });
const H2 = (t) => new D.Paragraph({ heading: D.HeadingLevel.HEADING_2, children: [run(t)] });
const bullet = (text) => new D.Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(text) ? text : [text]).map((p) => (typeof p === "string" ? run(p) : p)),
});
const num = (text) => new D.Paragraph({
  numbering: { reference: "num", level: 0 }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(text) ? text : [text]).map((p) => (typeof p === "string" ? run(p) : p)),
});
const code = (t) => new D.Paragraph({
  spacing: { after: 40 }, shading: { type: D.ShadingType.CLEAR, fill: "F2F2F2" },
  children: [new D.TextRun({ text: t, font: "Courier New", size: 18 })],
});

const border = { style: D.BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, header, rows, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (txt, w, head, bold) => new D.TableCell({
    width: { size: w, type: D.WidthType.DXA }, borders,
    shading: head ? { type: D.ShadingType.CLEAR, fill: "1F3A5F" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new D.Paragraph({ children: [run(String(txt), { size: o.size || 18, bold: head || bold, color: head ? "FFFFFF" : undefined })] })],
  });
  return new D.Table({
    width: { size: total, type: D.WidthType.DXA }, columnWidths: widths,
    rows: [
      new D.TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, widths[i], true)) }),
      ...rows.map((r) => new D.TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, widths[i], false, o.boldLast && i === 0)) })),
    ],
  });
}
async function figure(rel, widthPx, caption) {
  const file = path.join(ROOT, rel);
  const m = await sharp(file).metadata();
  const h = Math.round((widthPx * m.height) / m.width);
  return [
    new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new D.ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: h },
        altText: { title: caption, description: caption, name: path.basename(rel) } })] }),
    new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 200 },
      children: [run(caption, { italics: true, size: 18, color: "595959" })] }),
  ];
}
const gap = () => new D.Paragraph({ spacing: { after: 120 }, children: [] });
const pb = () => new D.Paragraph({ children: [new D.PageBreak()] });

function styles() {
  return {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "2E5C8A" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  };
}
const numbering = {
  config: [
    { reference: "bul", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ],
};
function footer(label) {
  return { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER, children: [
    run(label + "  |  page ", { size: 16, color: "7F7F7F" }),
    new D.TextRun({ children: [D.PageNumber.CURRENT], font: FONT, size: 16, color: "7F7F7F" })] })] }) };
}
function pack(children, label, file) {
  const doc = new D.Document({
    styles: styles(), numbering,
    sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1300, left: 1440, right: 1440 } } },
      footers: footer(label), children }],
  });
  return D.Packer.toBuffer(doc).then((b) => { fs.writeFileSync(file, b); console.log("wrote", file); });
}
module.exports = { D, R, ROOT, W, f2, f3, pm, run, P, B, I, H1, H2, bullet, num, code, table, figure, gap, pb, pack };
