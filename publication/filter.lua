-- Map the manuscript's semantic fenced divs to styled LaTeX environments so the
-- PDF matches the web edition: definition/theorem/.../remark/example/algorithm
-- become titled tcolorboxes (colours defined in preamble.tex), proofs become
-- amsthm proofs, and the inline role spans (box-title, wex-op, algo-lab, ex-tag,
-- qed) get their weight/marks back. Without this, Pandoc drops every wrapper and
-- the containers render as indistinguishable body text.

local box_env = {
  definition = "kbdef", theorem = "kbthm", lemma = "kbthm",
  proposition = "kbthm", corollary = "kbthm", remark = "kbrmk",
  example = "kbex", algorithm = "kbalgo",
}
local default_label = {
  definition = "Definition", theorem = "Theorem", lemma = "Lemma",
  proposition = "Proposition", corollary = "Corollary", remark = "Remark",
  example = "Example", algorithm = "Algorithm",
}

local function inlines_to_latex(inlines)
  return (pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "latex"):gsub("%s+$", ""))
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

function Div(el)
  for kind, env in pairs(box_env) do
    if el.classes:includes(kind) then
      local title = extract_title(el.content) or default_label[kind]
      local lbl = el.identifier ~= "" and ("\\label{" .. el.identifier .. "}") or ""
      table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{" .. env .. "}{" .. title .. "}" .. lbl))
      table.insert(el.content, pandoc.RawBlock("latex", "\\end{" .. env .. "}"))
      return el.content
    end
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
