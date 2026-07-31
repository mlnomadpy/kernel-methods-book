-- Map semantic fenced divs to plain LaTeX statement environments. The
-- environments preserve theorem/definition structure through typography and
-- whitespace only; they are intentionally not tcolorboxes.

local box_env = {
  definition = "kbdefinition", theorem = "kbtheorem", lemma = "kblemma",
  proposition = "kbproposition", corollary = "kbcorollary",
  remark = "kbremark", example = "kbexample", algorithm = "kbalgorithm",
}
local default_label = {
  definition = "Definition", theorem = "Theorem", lemma = "Lemma",
  proposition = "Proposition", corollary = "Corollary", remark = "Remark",
  example = "Example", algorithm = "Algorithm",
}

local function inlines_to_latex(inlines)
  return (pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "latex"):gsub("%s+$", ""))
end

-- Review-boilerplate paragraphs inside formal boxes ("**Assumptions.** ...",
-- "**Proof status.** ...", "**Verification artifact.** ...") are metadata, not
-- mathematics. Keep them, but set them small and muted so the statement stays
-- the visual subject of the box.
local meta_leads = {
  ["Assumptions."] = true, ["Proof status."] = true,
  ["Verification artifact."] = true,
}
local function demote_meta_paragraphs(blocks)
  for i, blk in ipairs(blocks) do
    if blk.t == "Para" and blk.content[1] and blk.content[1].t == "Strong" then
      local lead = pandoc.utils.stringify(blk.content[1])
      if meta_leads[lead] then
        blocks[i] = pandoc.Div({
          pandoc.RawBlock("latex", "\\kbmetaopen{}"),
          blk,
          pandoc.RawBlock("latex", "\\kbmetaclose{}"),
        })
      end
    end
  end
  return blocks
end

-- Pull a leading `[...]{.box-title}` span off the content and return it as a
-- LaTeX string (or nil). Drops the span, and the paragraph if it held nothing
-- else.
local function extract_title(blocks)
  local first = blocks[1]
  if first and (first.t == "Para" or first.t == "Plain") then
    for i, inl in ipairs(first.content) do
      if inl.t == "Span" and inl.classes:includes("box-title") then
        local title = inlines_to_latex(inl.content)
        first.content:remove(i)
        if pandoc.utils.stringify(first.content):gsub("%s", "") == "" then
          blocks:remove(1)
        end
        return title
      end
    end
  end
  return nil
end

local function extract_class_span(blocks, class_name)
  local first = blocks[1]
  if first and (first.t == "Para" or first.t == "Plain") then
    for i, inl in ipairs(first.content) do
      if inl.t == "Span" and inl.classes:includes(class_name) then
        local value = inlines_to_latex(inl.content)
        first.content:remove(i)
        if pandoc.utils.stringify(first.content):gsub("%s", "") == "" then
          blocks:remove(1)
        end
        return value
      end
    end
  end
  return nil
end

-- Manuscript titles conventionally read "Theorem (Representer theorem)".
-- amsthm already supplies "Theorem 3.7", so pass only the parenthesized name as
-- its optional note. A free-form title remains a note rather than being lost.
local function theorem_note(title, kind)
  if not title or title == "" or title == default_label[kind] then return nil end
  local prefix = default_label[kind]
  local note = title:match("^" .. prefix .. "%s*%((.*)%)$")
  if note then return note end
  note = title:match("^" .. prefix .. "%s*[:%.%-]%s*(.*)$")
  if note then return note end
  if title:match("^" .. prefix .. "%s*$") then return nil end
  return title
end

function Div(el)
  if el.classes:includes("code-listing") then
    local caption = extract_class_span(el.content, "listing-caption") or "Executable check"
    local number = el.attributes["data-number"] or ""
    local language = el.attributes["data-language"] or "Code"
    local identifier = el.identifier or ""
    table.insert(el.content, 1, pandoc.RawBlock("latex",
      "\\begin{kblisting}{" .. number .. "}{" .. caption .. "}{" .. language .. "}{" .. identifier .. "}"))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{kblisting}"))
    return el.content
  end

  for kind, env in pairs(box_env) do
    if el.classes:includes(kind) then
      local title = extract_title(el.content) or default_label[kind]
      local note = theorem_note(title, kind)
      local lbl = el.identifier ~= "" and ("\\label{" .. el.identifier .. "}") or ""
      el.content = demote_meta_paragraphs(el.content)
      local open = "\\begin{" .. env .. "}" .. (note and ("[" .. note .. "]") or "") .. lbl
      table.insert(el.content, 1, pandoc.RawBlock("latex", open))
      table.insert(el.content, pandoc.RawBlock("latex", "\\end{" .. env .. "}"))
      return el.content
    end
  end

  -- The chapter lead paragraph: a styled opener block (see kblead in the
  -- preamble). The build script emits it as `::: {.lead}` for the PDF.
  if el.classes:includes("lead") then
    table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{kblead}"))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{kblead}"))
    return el.content
  end

  if el.classes:includes("proof") then
    local title = extract_title(el.content)
    local opt = (title and title ~= "" and title ~= "Proof") and ("[" .. title .. "]") or ""
    table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{proof}" .. opt))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{proof}"))
    return el.content
  end

  if el.classes:includes("hint-body") then
    table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{kbhint}"))
    table.insert(el.content, pandoc.RawBlock("latex", "\\end{kbhint}"))
    return el.content
  end

  if el.classes:includes("wex-setup") then
    table.insert(el.content, 1, pandoc.RawBlock("latex", "\\noindent\\textit{\\color{kbEx}Setup.}\\enspace"))
    return el.content
  end

  -- Structural wrappers with no box of their own: unwrap to their content.
  if el.classes:includes("wex") or el.classes:includes("algo-io")
    or el.classes:includes("exercises") or el.classes:includes("tablewrap") then
    return el.content
  end
end

-- Runs bottom-up, i.e. before the enclosing Div. Do NOT touch `box-title` here:
-- the Div handler extracts it as the environment title. Drop `qed` (amsthm's
-- proof supplies the end-of-proof mark).
function Span(el)
  if el.classes:includes("box-title") then
    return nil
  end
  if el.classes:includes("qed") then
    return {}
  end
  if el.classes:includes("wex-op") or el.classes:includes("algo-lab") then
    return pandoc.Strong(el.content)
  end
  if el.classes:includes("ex-tag") then
    local out = { pandoc.RawInline("latex", "\\kbextag{") }
    for _, i in ipairs(el.content) do out[#out + 1] = i end
    out[#out + 1] = pandoc.RawInline("latex", "}")
    return out
  end
end
