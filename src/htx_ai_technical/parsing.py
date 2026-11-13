"""PDF -> ParsedDocument of section and table blocks, via Docling. Cached by PDF name.

Docling labels headings, paragraphs, tables, captions and page furniture, but returns
headings flat. The MOF template's numbering gives the hierarchy back:
"01 Update on ..." is a chapter, "1.2 Operating Revenue" a section, an unnumbered heading
a sub-heading. An unnumbered heading followed only by a "Table 2.1" label and a table is
that table's title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, DoclingDocument, TableItem

from .schemas import Block, ParsedDocument

# Items left out of the blocks: page headers and footers, pictures and the table of contents.
_SKIP = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE, DocItemLabel.DOCUMENT_INDEX}
# A table's number on a line of its own, e.g. "Table 2.1" or "Table 3.1a:".
_TABLE_LABEL = re.compile(r"^Table\s+\d+(\.\d+)?[a-z]?:?$", re.IGNORECASE)
# A chapter heading, e.g. "01 Update on ...".
_CHAPTER = re.compile(r"^\d{2}\s+\S")
# A numbered section heading, e.g. "1.2 Operating Revenue"; group 1 is the number.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\s+\S")
# The fiscal year on the cover, e.g. "Financial Year 2024".
_FISCAL_YEAR = re.compile(r"Financial\s+Year\s+(\d{4})", re.IGNORECASE)
# Level of an unnumbered heading: below a chapter (1) and a section like "1.2" (2).
_SUBHEADING = 3


def heading_level(text: str) -> int:
    """1 for a chapter, 2 for "1.2 ...", 3 for "1.2.3 ..." or an unnumbered sub-heading."""
    if _CHAPTER.match(text) or not re.search(r"[a-z]", text):  # "01 Update ..." or ALL CAPS
        return 1
    if m := _NUMBERED.match(text):
        return 1 + m.group(1).count(".")
    return _SUBHEADING


def _pages(item: DocItem) -> list[int]:
    return sorted({p.page_no for p in item.prov})


def _clean(text: str) -> str:
    """Collapse whitespace and drop the literal "BLANK" that the PDFs print in empty cells."""
    return " ".join(text.replace("BLANK", " ").split())


def _is_note(item: DocItem, text: str) -> bool:
    """A table's footnote: "Note: ...", "Source: ..." or a line starting with a footnote number."""
    return item.label == DocItemLabel.FOOTNOTE or bool(re.match(r"^(Note|Source)s?\b|^\d+\s", text))


def table_grid(table: TableItem) -> tuple[list[list[str]], int]:
    """Cleaned cell grid and the number of leading header rows."""
    grid = [[_clean(c.text) for c in row] for row in table.data.grid]
    header_rows = 0
    for row in table.data.grid:
        if not any(c.column_header for c in row):
            break
        header_rows += 1
    keep = [i for i, row in enumerate(grid) if any(row)]  # drop all-empty rows
    header_rows = sum(i < header_rows for i in keep)  # recount: some header rows may have been dropped
    return [grid[i] for i in keep], header_rows


def grid_to_markdown(grid: list[list[str]], header_rows: int) -> str:
    """A markdown table. Stacked header rows are joined into one header per column."""
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    rows = [r + [""] * (width - len(r)) for r in grid]
    head = [" ".join(dict.fromkeys(rows[i][j] for i in range(header_rows) if rows[i][j])) for j in range(width)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[header_rows:]]
    return "\n".join(lines)


@dataclass
class _Builder:
    """Groups Docling items, fed in reading order, into section and table blocks."""

    blocks: list[Block] = field(default_factory=list)
    stack: list[tuple[int, str]] = field(default_factory=list)  # open headings (level, title)
    section: Block | None = None  # section being filled
    table: Block | None = None  # last table, receives its notes and page-continuations
    table_label: str | None = None  # pending "Table 2.1" label
    since_table: bool = False  # a heading or label appeared since the last table

    def _path(self) -> list[str]:
        return [title for _, title in self.stack]

    def _flush(self) -> None:
        if self.section and self.section.text.strip():
            self.blocks.append(self.section)
        self.section = None

    def heading(self, text: str, pages: list[int]) -> None:
        self._flush()
        level = heading_level(text)
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        path = self._path()
        self.stack.append((level, text))
        self.section = Block(id="", type="section", title=text, section_path=path, pages=pages, text="")
        self.table, self.since_table = None, True

    def paragraph(self, item: DocItem, text: str) -> None:
        if _TABLE_LABEL.match(text):
            self.table_label, self.since_table = text.rstrip(":"), True
            return
        if self.table is not None and _is_note(item, text):
            self.table.text += "\n" + text
            self.table.pages = sorted({*self.table.pages, *_pages(item)})
            return
        if self.section is None:  # text before any heading, or resuming after a table
            path = self._path()
            title = path[-1] if path else "(front matter)"
            self.section = Block(id="", type="section", title=title, section_path=path[:-1], pages=[], text="")
        self.section.text = f"{self.section.text}\n{text}".strip()
        self.section.pages = sorted({*self.section.pages, *_pages(item)})
        self.table = None

    def add_table(self, table: TableItem, doc: DoclingDocument) -> None:
        grid, header_rows = table_grid(table)
        pages = _pages(table)
        caption = _clean(table.caption_text(doc))

        # Page continuation: untitled table right after a table with the same width.
        prev = self.table
        if not caption and not self.since_table and prev and prev.grid and len(prev.grid[0]) == len(grid[0]):
            body = grid[header_rows:] if grid[:header_rows] == prev.grid[: prev.header_rows] else grid
            prev.grid += body
            prev.pages = sorted({*prev.pages, *pages})
            prev.text = grid_to_markdown(prev.grid, prev.header_rows)
            return

        title = caption
        # No caption from Docling: build a title from what came just before the table.
        if not title:
            # The open section is a heading with no text under it yet.
            heading = self.section if self.section and not self.section.text.strip() else None
            # It has no chapter ("01 ...") or section ("1.2 ...") number, so it names the
            # table rather than a real section: take it as the title and drop it from the
            # heading stack, so the text after the table goes back to the parent section.
            if heading and not _CHAPTER.match(heading.title) and not _NUMBERED.match(heading.title):
                title = heading.title
                self.stack.pop()
                self.section = None
            # A "Table 2.1" line was seen and isn't already in the title: prefix it.
            if self.table_label and self.table_label.lower() not in title.lower():
                title = f"{self.table_label} {title}".strip()
            # Still nothing: fall back to "Table in <current heading>".
            title = title or f"Table in {self._path()[-1] if self.stack else 'document'}"
        self._flush()
        block = Block(
            id="",
            type="table",
            title=title,
            section_path=[t for level, t in self.stack if level == 1],
            pages=pages,
            text=grid_to_markdown(grid, header_rows),
            grid=grid,
            header_rows=header_rows,
        )
        self.blocks.append(block)
        self.table, self.table_label, self.since_table = block, None, False

    def finish(self) -> list[Block]:
        self._flush()
        return self.blocks


def build_blocks(doc: DoclingDocument) -> list[Block]:
    """Walk Docling items in reading order and group them into section and table blocks."""
    captions = {
        ref.cref for item, _ in doc.iterate_items() for ref in getattr(item, "captions", []) or []
    }
    b = _Builder()
    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem) or item.label in _SKIP or item.self_ref in captions:
            continue
        if isinstance(item, TableItem):
            b.add_table(item, doc)
            continue
        text = _clean(getattr(item, "text", "") or "")
        if not text:
            continue
        if item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            b.heading(text, _pages(item))
        else:
            b.paragraph(item, text)
    return b.finish()


def cover_block(doc: DoclingDocument) -> Block:
    """Every text item on page 1, page furniture included: Docling files the cover's
    "Distributed on Budget Day: ..." line as a page footer, and later parts need it."""
    lines = [
        _clean(getattr(item, "text", "") or "")
        for item, _ in doc.iterate_items(page_no=1, included_content_layers=set(ContentLayer))
        if any(p.page_no == 1 for p in getattr(item, "prov", []))
    ]
    return Block(id="", type="section", title="(cover)", pages=[1], text="\n".join(l for l in lines if l))


def detect_fiscal_year(first_page: str) -> int | None:
    """The fiscal year printed on the first page, e.g. "Financial Year 2024"."""
    m = _FISCAL_YEAR.search(first_page)
    return int(m.group(1)) if m else None


def convert_pdf(pdf_path: Path) -> DoclingDocument:
    # Imported here because Docling is slow to import and a cached run doesn't need it.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TableStructureOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    table_options = TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=True)
    options = PdfPipelineOptions(
        do_ocr=False,  # MOF PDFs are born-digital
        do_table_structure=True,
        table_structure_options=table_options,
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return converter.convert(pdf_path).document


def load_document(pdf_path: Path, cache_dir: Path) -> ParsedDocument:
    """Parse `pdf_path` into blocks, reusing `<cache_dir>/<pdf-name>.json` whenever it exists.
    Delete that file to re-parse."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    cache = cache_dir / f"{pdf_path.stem}.json"
    if cache.is_file():
        return ParsedDocument.model_validate_json(cache.read_text())

    docling_doc = convert_pdf(pdf_path)
    cover = cover_block(docling_doc)
    blocks = [cover, *build_blocks(docling_doc)]
    for i, b in enumerate(blocks):
        b.id = f"{b.type[0]}{i:03d}"  # e.g. "s000" for the cover, "t026" for a table
    parsed = ParsedDocument(source=pdf_path.name, fiscal_year=detect_fiscal_year(cover.text), blocks=blocks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(parsed.model_dump_json(indent=1))
    return parsed
