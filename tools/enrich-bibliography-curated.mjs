#!/usr/bin/env node
/** Add title-verified links from official venue and archival pages. */
import fs from "node:fs";

const write = process.argv.includes("--write");
const bibPath = "bibliography.bib";
const source = fs.readFileSync(bibPath, "utf8");
const links = new Map(Object.entries({
  berg1984: { doi: "10.1007/978-1-4612-1128-0" },
  broomhead1988: "https://www.complex-systems.com/abstracts/v02_i03_a05/",
  cravenwahba1979: { doi: "10.1007/BF01404567" },
  dhillon2004: { doi: "10.1145/1014052.1014118" },
  dicker2015: { doi: "10.1214/17-EJS1258" },
  friess1998: "https://dblp.org/rec/conf/icml/FriessCC98",
  grunewalder2012: "https://icml.cc/2012/papers/898.pdf",
  haussler1999: "https://stemcellgenomics.ucsc.edu/miscellaneous-texts/",
  hensman2013bigdata: "https://www.microsoft.com/en-us/research/?p=545478",
  herglotz1911: "https://archiv.saw-leipzig.de/saw-archive/publikationen-quellen/publikationen/berichte-ueber-die-verhandlungen-der-koeniglich-saechsischen-gesellschaft-der-wissenschaften-zu-leipzig-mathematisch-physische-klasse-bd-1/berichte-ueber-die-verhandlungen-der-koeniglich-saechsischen-gesellschaft-der-wissenschaften-zu-leipzig-mathematisch-physische-klasse-bd-63/ueber-potenzreihen-mit-positivem-reellen-teil-im-einheitskreis",
  hsu2002: { doi: "10.1109/72.991427" },
  huszar2012: "https://auai.org/uai2012/papers/213.pdf",
  joachims1999: { doi: "10.17877/DE290R-14262" },
  kantorovich1942: "https://www.mathnet.ru/eng/znsl/v312/p11",
  kashima2003: "https://www.ed.aaai.org/Library/ICML/2003/icml03-044.php",
  kin2002: "https://pubmed.ncbi.nlm.nih.gov/14571380/",
  kipf2017: "https://openreview.net/forum?id=SJU4ayYgl",
  kloft2011: "https://www.jmlr.org/papers/v12/kloft11a.html",
  kolmogorov1941: "https://books.google.com/books?id=-6krAAAAYAAJ",
  kondor2002: "https://dblp.org/rec/conf/icml/KondorL02",
  lanckriet2004: "https://jmlr.org/papers/v5/lanckriet04a.html",
  le2013fastfood: "https://proceedings.mlr.press/v28/le13.html",
  lecun1995: "https://yann.lecun.com/exdb/publis/",
  ledoux1991: { doi: "10.1007/978-3-642-20212-4" },
  lee2018nngp: "https://openreview.net/forum?id=B1EA-M-0Z",
  li2020fno: "https://openreview.net/forum?id=c8P9NQVtmnO",
  linlin2003: "https://www.csie.ntu.edu.tw/~htlin/paper/",
  logan2001: "https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/Compaq-DEC/index.html",
  loosli2016krein: { doi: "10.1109/TPAMI.2015.2477830" },
  matern1960: { doi: "10.1007/978-1-4615-7892-5" },
  matthews2018: "https://openreview.net/forum?id=H1-nGgWC-",
  nesterov2004: { doi: "10.1007/978-1-4419-8853-9" },
  neyshabur2018: "https://arxiv.org/abs/1707.09564",
  oglic2018: "https://proceedings.mlr.press/v80/oglic18a.html",
  pekalska2005: { doi: "10.1142/5965" },
  platt1998: "https://www.microsoft.com/en-us/research/?p=152322",
  platt2000: "https://mitpress.mit.edu/9780262194488/advances-in-large-margin-classifiers/",
  ramon2003: "https://publica.fraunhofer.de/entities/publication/e5c88bb6-e5a8-4ce9-8b50-9a52d07899d6",
  ramsay2005fda: { doi: "10.1007/b98888" },
  rifkin2004: "https://www.jmlr.org/papers/v5/rifkin04a.html",
  rifkin2007rls: "https://www.mit.edu/~9.520/fall16/Classes/rls.html",
  saunders1998: "https://dblp.org/rec/conf/icml/SaundersGV98.html",
  scholkopf1997: "https://dblp.org/rec/phd/dnb/Scholkopf97",
  shalevshwartz2014book: { doi: "10.1017/CBO9781107298019" },
  song2013cme: { doi: "10.1109/MSP.2013.2252713" },
  srinivas2010: "https://arxiv.org/abs/0912.3995",
  smola2000: "https://is.mpg.de/ei/publications/819",
  steinwart2009rates: "https://www.cs.mcgill.ca/~colt2009/papers/038.pdf",
  sutherland2017: "https://openreview.net/forum?id=HJWHIKqgl",
  tsai2019transformer: { doi: "10.18653/v1/D19-1443" },
  vapnik1982: { doi: "10.1007/978-1-4757-2440-0" },
  villani2009: { doi: "10.1007/978-3-540-71050-9" },
  williams1996: "https://proceedings.neurips.cc/paper/1995/hash/7cce53cf90577442771720a370c3c723-Abstract.html",
  wilson2013: "https://proceedings.mlr.press/v28/wilson13.html",
  weisfeiler1968: "https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf",
  xu2019gin: "https://openreview.net/forum?id=ryGs6iA5Km",
  zhang2011kcit: "https://auai.org/uai2011/accepted.html",
}));

const entryRe = /@(misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi;
let enriched = 0;
const updated = source.replace(entryRe, (whole, type, key, block) => {
  const value = links.get(key);
  if (!value) return whole;
  const record = typeof value === "string" ? { url: value } : value;
  const hasDoi = /^\s*doi\s*=/mi.test(block);
  const hasUrl = /^\s*url\s*=/mi.test(block);
  const additions = [];
  if (record.doi && !hasDoi) additions.push(`  doi = {${record.doi}}`);
  if (!hasUrl) additions.push(`  url = {${record.url || `https://doi.org/${record.doi}`}}`);
  if (!additions.length) return whole;
  enriched += 1;
  const trimmed = block.replace(/\s+$/, "").replace(/,?$/, ",");
  return `@${type}{${key},${trimmed}\n${additions.join(",\n")}\n}`;
});

console.log(`${enriched}/${links.size} curated authoritative links available for enrichment.`);
if (write) {
  fs.writeFileSync(bibPath, updated);
  console.log("Updated bibliography.bib");
} else {
  console.log("Dry run only; pass --write to update bibliography.bib");
}
