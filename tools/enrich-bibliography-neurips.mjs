#!/usr/bin/env node
/** Add title-verified links from the official NeurIPS proceedings index. */
import fs from "node:fs";

const write = process.argv.includes("--write");
const bibPath = "bibliography.bib";
const source = fs.readFileSync(bibPath, "utf8");
const links = new Map(Object.entries({
  chen2019rkn: "https://proceedings.neurips.cc/paper_files/paper/2019/file/d60743aab4b625940d39b3b51c3c6a78-Paper.pdf",
  drucker1997svr: "https://proceedings.neurips.cc/paper_files/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html",
  gorham2015: "https://proceedings.neurips.cc/paper/2015/hash/698d51a19d8a121ce581499d7b701668-Abstract.html",
  gretton2008hsictest: "https://proceedings.neurips.cc/paper_files/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html",
  gretton2009fast: "https://proceedings.neurips.cc/paper/2009/hash/9246444d94f081e3549803b928260f56-Abstract.html",
  gretton2012optkernel: "https://proceedings.neurips.cc/paper/2012/hash/dbe272bab69f8e13f14b405e038deb64-Abstract.html",
  haochen2021: "https://proceedings.neurips.cc/paper_files/paper/2021/hash/27debb435021eb68b3965290b5e24c49-Abstract.html",
  jaakkola1999: "https://proceedings.neurips.cc/paper/1998/hash/db1915052d15f7815c8b88e879465a1e-Abstract.html",
  jaakkola1999fisher: "https://proceedings.neurips.cc/paper/1998/hash/db1915052d15f7815c8b88e879465a1e-Abstract.html",
  jacot2018: "https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html",
  jitkrittum2016: "https://proceedings.neurips.cc/paper/2016/hash/0a09c8844ba8f0936c20bd791130d6b6-Abstract.html",
  johnson2013: "https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html",
  lin2015catalyst: "https://proceedings.neurips.cc/paper/2015/hash/c164bbc9d6c72a52c599bbb43d8db8e1-Abstract.html",
  liu2016svgd: "https://proceedings.neurips.cc/paper/2016/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html",
  ma2017eigenpro: "https://proceedings.neurips.cc/paper/2017/hash/bf424cb7b0dea050a42b9739eb261a3a-Abstract.html",
  mairal2016: "https://proceedings.neurips.cc/paper_files/paper/2016/hash/fc8001f834f6a5f0561080d134d53d29-Abstract.html",
  meanti2020: "https://proceedings.neurips.cc/paper/2020/hash/a59afb1b7d82ec353921a55c579ee26d-Abstract.html",
  mika1999: "https://proceedings.neurips.cc/paper/1998/hash/226d1f15ecd35f784d2a20c3ecf56d7f-Abstract.html",
  mikolov2013word2vec: "https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html",
  montufar2014: "https://proceedings.neurips.cc/paper_files/paper/2014/hash/fa6f2a469cc4d61a92d96e74617c3d2a-Abstract.html",
  muandet2012smm: "https://proceedings.neurips.cc/paper/2012/hash/9bf31c7ff062936a96d3c8bd1f8f2ff3-Abstract.html",
  muandet2020dualiv: "https://proceedings.neurips.cc/paper_files/paper/2020/hash/1c383cd30b7c298ab50293adfecb7b18-Abstract.html",
  ng2002: "https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html",
  paciorek2004: "https://proceedings.neurips.cc/paper/2003/hash/326a8c055c0d04f5b06544665d8bb3ea-Abstract.html",
  rahimi2007: "https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html",
  rasmussen2003bmc: "https://proceedings.neurips.cc/paper_files/paper/2002/hash/24917db15c4e37e421866448c9ab23d8-Abstract.html",
  rudi2015: "https://proceedings.neurips.cc/paper_files/paper/2015/hash/03e0704b5690a2dee1861dc3ad3316c9-Abstract.html",
  rudi2017: "https://proceedings.neurips.cc/paper_files/paper/2017/hash/61b1fb3f59e28c67f3925f3c79be81a1-Abstract.html",
  rudi2017falkon: "https://proceedings.neurips.cc/paper/2017/hash/05546b0e38ab9175cd905eebcc6ebb76-Abstract.html",
  seeger2002: "https://proceedings.neurips.cc/paper/2001/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf",
  sejdinovic2013interaction: "https://proceedings.neurips.cc/paper/2013/hash/076a0c97d09cf1a0ec3e19c7f2529f2b-Abstract.html",
  singh2019kiv: "https://proceedings.neurips.cc/paper_files/paper/2019/hash/17b3c7061788dbe82de5abe9f6fe22b3-Abstract.html",
  snelson2006fitc: "https://proceedings.neurips.cc/paper_files/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html",
  snoek2012: "https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html",
  tipping2001sparse: "https://proceedings.neurips.cc/paper/2000/hash/bf201d5407a6509fa536afc4b380577e-Abstract.html",
  togninalli2019wwl: "https://proceedings.neurips.cc/paper/2019/hash/73fed7fd472e502d8908794430511f4d-Abstract.html",
  williams2001: "https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html",
  yang2019tp: "https://proceedings.neurips.cc/paper/2019/hash/5e69fda38cda2060819766569fd93aa5-Abstract.html",
  yu2016orf: "https://proceedings.neurips.cc/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html",
}));

const entryRe = /@(misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi;
let enriched = 0;
const updated = source.replace(entryRe, (whole, type, key, block) => {
  const url = links.get(key);
  if (!url || /^\s*(?:url|doi)\s*=/mi.test(block)) return whole;
  enriched += 1;
  const trimmed = block.replace(/\s+$/, "").replace(/,?$/, ",");
  return `@${type}{${key},${trimmed}\n  url = {${url}}\n}`;
});

console.log(`${enriched}/${links.size} title-verified NeurIPS links available for enrichment.`);
if (write) {
  fs.writeFileSync(bibPath, updated);
  console.log("Updated bibliography.bib");
} else {
  console.log("Dry run only; pass --write to update bibliography.bib");
}
