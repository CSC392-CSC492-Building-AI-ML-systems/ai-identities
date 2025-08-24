'use client';
import { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from '@/lib/muiTheme';
import SearchIcon from '@mui/icons-material/Search';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField,
  Typography, FormGroup, FormControlLabel, Checkbox, Chip,
  Button, Box, Paper, InputBase, IconButton, ToggleButtonGroup, ToggleButton
} from '@mui/material';
import { useParams, useRouter } from 'next/navigation';
import { XWIKI_URL } from "@/constants"

export interface XWikiClass {
  links: XWikiLink[];
  properties: XWikiProperty[];
}

export interface XWikiLink {
  href: string;
  rel: string;
  type: string | null;
  hrefLang: string | null;
}

export interface XWikiProperty {
  links: XWikiLink[];
  attributes: XWikiAttribute[];
  value: string | null;
  name: string;
  type: string;
}

export interface XWikiAttribute {
  links: XWikiLink[];
  name: string;
  value: string;
}

type TagMap = { [group: string]: string[] };


const MODES = {
  llms: {
    label: 'LLM Pages',
    className: 'XWiki.LLMClass',
    nameProp: 'llms',
    url: 'XWiki/LLMClass',
    tagGroups: ["inputModalities", "outputModalities", "miscTags", "languages"],
    textFilterFields: ["predecessor", "successor", "modelSeries", "creators", "baseModel"]
  },
  llm_apps: {
    label: 'LLM Apps Pages',
    className: 'XWiki.LLMAppClass',
    nameProp: 'llm apps',
    url: 'XWiki/LLMAppClass',
    tagGroups: ["uses"],
    textFilterFields: ["creators", "llmsUsed"]
  },
};
type ModeKey = keyof typeof MODES;

const EXTRA_TAG_FIELDS: Record<ModeKey, string[]> = {
  llms: ["inputModalities", "outputModalities", "miscTags", "languages"],
  llm_apps: ['useCases', 'limitations', 'risks', 'llms', 'llmsGuess'],
};

// helper functions to display search results
const propsToMap = (r: any) => {
  const map: Record<string, any> = {};
  (r.object?.properties || []).forEach((p: any) => {
    map[p.name?.trim()] = p.value;
  });
  return map;
};

const splitTags = (s?: string) =>
  (s || '')
    .split(',')
    .map(t => t.trim().toLowerCase())
    .filter(Boolean);


export default function HomePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTags, setSelectedTags] = useState<{ [group: string]: string[] }>({});


  // Date filter state
  const [dateFilter, setDateFilter] = useState<{
    startDate: string;
    endDate: string;
  }>({
    startDate: '',
    endDate: ''
  });


  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Dialog state for filters
  const [filtersOpen, setFiltersOpen] = useState(false);

  // add state
  const [mode, setMode] = useState<ModeKey>('llms');
  const [didSearch, setDidSearch] = useState<Boolean>(false);

  // keep track of link clicked
  const [curWikiPage, setCurWikiPage] = useState<string | null>(null);

  const [fieldPrettyNames, setFieldPrettyNames] = useState<{ [group: string]: string }>({});
  const params = useParams();
  const slug = params?.slug as string[] | undefined;
  const router = useRouter()

  const [availableTags, setAvailableTags] = useState<{ [group: string]: string[] }>(Object.fromEntries(
    MODES[mode].tagGroups.map(i => [i, []])
  ));

  const [textFilterFields, setTextFilterFields] = useState<{ [group: string]: string }>(Object.fromEntries(
    MODES[mode].textFilterFields.map(i => [i, ''])
  ));

  console.log(slug)
  console.log(curWikiPage)


  const searchHelper = (searchTerm: string, fieldName: string) => {
    if (searchTerm.includes('\"')) {
      console.log(searchTerm)
      searchTerm = searchTerm.slice(1, -1)
      return `(lower(obj.${fieldName}) LIKE lower('%|${searchTerm}|%') OR lower(obj.${fieldName}) LIKE lower('%|${searchTerm}') OR lower(obj.${fieldName}) LIKE lower('${searchTerm}|%') OR lower(obj.${fieldName}) LIKE lower('${searchTerm}'))`
    } else {
      if (MODES[mode].textFilterFields.includes(fieldName)) {
        return `(lower(obj.${fieldName}) LIKE lower('%|${searchTerm}%') OR lower(obj.${fieldName}) LIKE lower('${searchTerm}%'))`
      } else {
        return `lower(obj.${fieldName}) LIKE lower('%${searchTerm}%')`
      }

    }
  }

  useEffect(() => {
    async function test() {
      console.log(await (await fetch("https://wiki.llm.test/bin/view/XWiki/LLMClass")).text())
    }
    test()
    if (slug) {
      setCurWikiPage(`${XWIKI_URL}/bin/view/XWiki/${slug.join('/')}`)
    }
  }, [])

  useEffect(() => {
    if (!curWikiPage) {
      router.replace('/search',)
    } else {
      console.log("hi")
      router.replace(`/search/${curWikiPage.split("/").slice(-2).join('/')}`)
    }
  }, [curWikiPage])

  // reset tag containers when mode changes (so UI doesn't show old groups)
  useEffect(() => {
    const tagGroups = Object.fromEntries(MODES[mode].tagGroups.map(g => [g, []]));
    const textGroups = Object.fromEntries(MODES[mode].textFilterFields.map(g => [g, '']));
    setSelectedTags(tagGroups as TagMap);
    setAvailableTags(tagGroups as TagMap);
    setTextFilterFields(textGroups)
    setResults([]);
    setError(null);

    setSearchTerm('');
    setDidSearch(false);

    // Reset date filter when mode changes
    setDateFilter({ startDate: '', endDate: '' });
  }, [mode]);

  // Load tags from XWiki
  useEffect(() => {
    const fetchTags = async () => {
      const { className, tagGroups } = MODES[mode];


      const url = `${XWIKI_URL}/rest/wikis/xwiki/classes/${className}/properties?media=json`;
      console.log(url)
      try {
        const res = await fetch(url, { headers: { 'Accept': 'application/json', "Authorization": "Basic " + btoa('ahmed33033:ahmed2003') } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: XWikiClass = await res.json();

        const grouped: { [key: string]: Set<string> } =
          Object.fromEntries(tagGroups.map(g => [g, new Set<string>()]));

        let prettyNames: { [key: string]: string } = {}

        data.properties.forEach(result => {
          console.log(result)
          prettyNames[result.name] = result.attributes[1].value
          if (MODES[mode].tagGroups.includes(result.name)) {
            result.attributes.at(-1)?.value?.split('|').forEach((tag: string) => {
              const t = tag.trim().toLowerCase();
              if (t) grouped[result.name].add(t);
            });
          }
        });

        setFieldPrettyNames(prettyNames)

        const flat: TagMap = {};
        for (const g of tagGroups) flat[g] = Array.from(grouped[g] || []).sort((a, b) => a.localeCompare(b));
        setAvailableTags(flat);
      } catch (err) {
        console.error("Failed to fetch tags:", err);
      }
    };

    fetchTags();
  }, [mode]);

  const handleToggle = (group: string, value: string) => {
    setSelectedTags((prev) => {
      const current = prev[group] || [];
      const next = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return { ...prev, [group]: next };
    });
  };
  /* ------------------------------
              SEARCH
  ------------------------------ */
  const searchPages = async () => {
    const { className } = MODES[mode];

    let query = `doc.object(${className}) as obj`

    let queryArr: string[] = ["lower(doc.name) NOT LIKE lower('%template%')"]

    if (searchTerm) {
      queryArr.push(searchHelper(searchTerm, "name"))
    }


    for (const i of Object.keys(selectedTags)) {
      if (selectedTags[i].length) {
        queryArr.push(selectedTags[i].map((entry) => { return searchHelper(entry, i) }).join(' AND '))
      }
    }

    for (const i of Object.keys(textFilterFields)) {
      if (textFilterFields[i]) {
        queryArr.push(textFilterFields[i].split(",").map((entry) => { return searchHelper(entry, i) }).join(' AND '))
      }
    }

    if (dateFilter.startDate) {
      console.log(dateFilter.startDate)
      queryArr.push(`obj.releaseDate > '${dateFilter.startDate}'`)
    }

    if (dateFilter.endDate) {
      queryArr.push(`obj.releaseDate < '${dateFilter.endDate}'`)
    }
    query += ' where ' + queryArr.join(' AND ')


    console.log(query)

    console.log(encodeURIComponent(query))
    const fullUrl = `${XWIKI_URL}/rest/wikis/xwiki/query` +
      "?q=," + encodeURIComponent(query) +
      `&type=xwql&media=json&number=100&className=${encodeURIComponent(className)}`;
    console.log(fullUrl)
    try {
      const res = await fetch(fullUrl, { headers: { "Accept": "application/json", "Authorization": "Basic " + btoa("ahmed33033:ahmed2003") } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setResults(data.searchResults || []);

      setDidSearch(true);
      setError(null);
    } catch (err: any) {
      console.log("wa")
      console.log(err?.message)
      setError("Failed to fetch search data: " + (err?.message || String(err)));
      setResults([]);
    } finally {
      setDidSearch(true);
      setCurWikiPage(null);
    }
  };

  // ---- Filters dialog helpers ----
  const openFilters = () => setFiltersOpen(true);
  const closeFilters = () => setFiltersOpen(false);
  const clearGroup = (group: string) =>
    setSelectedTags(prev => ({ ...prev, [group]: [] }));
  const clearAll = () => {
    setSelectedTags({});
    setTextFilterFields(Object.fromEntries(MODES[mode].textFilterFields.map(i => [i, ''])));
    setDateFilter({ startDate: '', endDate: '' });
  };

  // Helper to check if date filter is active
  const hasActiveDateFilter = dateFilter.startDate || dateFilter.endDate;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <Box sx={{ maxWidth: 1200, margin: '2rem auto', padding: '1rem', paddingTop: '80px' }}>
        {/* Search Bar + Filters Button */}
        <Box sx={{
          display: 'flex', gap: 2, mb: 2, alignItems: 'center', justifyContent: 'center', flexDirection: {
            xs: "column", lg: "row"
          }
        }} >
          <Paper variant="outlined" sx={{ width: '60%', display: 'flex', alignItems: 'center', justifyContent: 'flex-start', px: 2, py: 0.5, gap: 1 }}>
            <ToggleButtonGroup size="small" exclusive value={mode}
              sx={{
                border: '1px solid', borderColor: 'divider', borderRadius: 1, overflow: 'hidden',
                '& .MuiToggleButton-root': { border: 0, px: 1.25, py: 0.25, fontSize: 12 },
                '& .Mui-selected': { bgcolor: 'action.selected' },
              }}
              onChange={(_, v) => { if (v) { setMode(v as ModeKey); setResults([]); setError(null); } }}
            >
              <ToggleButton value="llms">LLMs</ToggleButton>
              <ToggleButton value="llm_apps">Apps</ToggleButton>
            </ToggleButtonGroup>

            <InputBase
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); searchPages(); }
              }}
              sx={{ ml: 1, flex: 1 }}
              placeholder={`Search by ${MODES[mode].nameProp}`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <IconButton onClick={searchPages}><SearchIcon /></IconButton>
          </Paper>
          <Box sx={{ columnGap: "15px", display: "flex" }}>
            {!curWikiPage ?
              <>
                <Button variant="outlined" onClick={openFilters}>Filters</Button>
                <Button variant="contained" onClick={searchPages}>Search</Button>
              </>
              :
              <Button variant="contained" onClick={() => {
                setCurWikiPage(null)

              }}>Back to searching</Button>
            }
          </Box>
        </Box>

        {/* Selected Filter Chips */}
        <Box sx={{ mt: 2 }}>
          {Object.entries(selectedTags).flatMap(([group, tags]) =>
            (tags || []).map((tag) => (
              <Chip
                key={`${group}-${tag}`}
                label={`${fieldPrettyNames[group]}: ${tag}`}
                onDelete={() => handleToggle(group, tag)}
                sx={{ mr: 1, mb: 1 }}
              />
            ))
          )}
          {Object.entries(textFilterFields).map(([field, value]) =>
            value.trim() ? (
              <Chip
                key={`text-${field}`}
                label={`${fieldPrettyNames[field]}: ${value}`}
                onDelete={() => setTextFilterFields(prev => ({ ...prev, [field]: '' }))}
                sx={{ mr: 1, mb: 1 }}
              />
            ) : null
          )}
          {hasActiveDateFilter && (
            <Chip
              label={`Release Date: ${dateFilter.startDate || 'Any'} - ${dateFilter.endDate || 'Any'}`}
              onDelete={() => setDateFilter({ startDate: '', endDate: '' })}
              sx={{ mr: 1, mb: 1 }}
            />
          )}
        </Box>

        {/* Error Message */}
        {error && <pre style={{ color: 'red' }}>{error}</pre>}

        {/* Search Results */}
        {/* display the xwiki page using an iframe*/}
        {curWikiPage !== null ? (
          <div className="w-screen absolute left-0 flex-1 flex justify-center">
            <iframe
              src={curWikiPage}
              title={`Wiki: ${curWikiPage}`}
              className="w-97/100 h-[80vh] "
              style={{ minHeight: 400 }}
            />
          </div>
        ) : /* display the search results otherwise */ (
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 2 }}>
            {results.length === 0 ? ( // check if the user did a search with results
              didSearch === false || error !== null ? (
                <Typography variant="body2" sx={{ opacity: 0.8 }}></Typography>
              ) : (
                <Typography variant="body2" sx={{ opacity: 0.8 }}>No results found.</Typography>)
            ) : (
              results.map((r, index) => {
                const props = propsToMap(r);

                // Name / title
                const name =
                  r.title ||
                  props.name ||
                  r.pageName ||
                  r.space ||
                  'Untitled';

                // Creator + Release date
                console.log(props)
                const creator = props.creators || 'Unknown';

                const release =
                  props.releaseDate
                    ? new Date(props.releaseDate).toLocaleDateString()
                    : undefined;

                // collect tag chips (groups + a couple mode-specific extras)
                const tagFields = Array.from(
                  new Set([...(MODES[mode].tagGroups as readonly string[]), ...EXTRA_TAG_FIELDS[mode]])
                );

                const chips = tagFields.flatMap((field) =>
                  splitTags(String(props[field] ?? '')).map((t) => ({ field, t }))
                );

                // view URL
                const appName = name;
                const encodedName = encodeURIComponent(String(appName).trim());

                // display box for each result
                return (
                  <Box
                    key={index}
                    sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 2, cursor: "pointer" }}
                    onClick={() => setCurWikiPage(`${XWIKI_URL}/bin/view/${MODES[mode].url}/${encodedName}`)}>
                    <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="h6" sx={{ mr: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {name}
                      </Typography>
                    </Box>

                    <Typography variant="body2" sx={{ mb: 1.5 }}>
                      Creator: <strong>{creator}</strong>
                      {release ? <> • Released: <strong>{release}</strong></> : null}
                    </Typography>

                    {chips.length > 0 && (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                        {chips.map(({ field, t }, i) => (
                          <Chip key={`${field}-${t}-${i}`} label={`${fieldPrettyNames[field]}: ${t}`} size="small" />
                        ))}
                      </Box>
                    )}
                  </Box>
                );
              })
            )}
          </Box>
        )}
      </Box>

      {/* Filters Dialog */}
      <Dialog open={filtersOpen} onClose={closeFilters} fullWidth maxWidth="md" style={{ overflowY: "auto" }}>
        <DialogTitle>Filters</DialogTitle>
        <DialogContent dividers style={{ overflowY: "auto" }}>
          {/* Live-selected chips */}
          <Box sx={{ mb: 2 }}>
            {Object.entries(selectedTags).flatMap(([group, tags]) =>
              (tags || []).map((tag) => (
                <Chip
                  key={`${group}-${tag}`}
                  label={`${group}: ${tag}`}
                  onDelete={() => handleToggle(group, tag)}  // live toggle
                  sx={{ mr: 1, mb: 1 }}
                />
              ))
            )}
            {Object.entries(textFilterFields).map(([field, value]) =>
              value.trim() ? (
                <Chip
                  key={`text-${field}`}
                  label={`${field}: ${value}`}
                  onDelete={() => setTextFilterFields(prev => ({ ...prev, [field]: '' }))}
                  sx={{ mr: 1, mb: 1 }}
                />
              ) : null
            )}
            {hasActiveDateFilter && (
              <Chip
                label={`Release Date: ${dateFilter.startDate || 'Any'} - ${dateFilter.endDate || 'Any'}`}
                onDelete={() => setDateFilter({ startDate: '', endDate: '' })}
                sx={{ mr: 1, mb: 1 }}
              />
            )}
          </Box>

          {/* Date Filter Section */}
          <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>Release Date Filter</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <TextField
                label="From Date"
                type="date"
                value={dateFilter.startDate}
                onChange={(e) => setDateFilter(prev => ({ ...prev, startDate: e.target.value }))}
                InputLabelProps={{ shrink: true }}
                sx={{
                  minWidth: 150,
                  '& input[type="date"]::-webkit-calendar-picker-indicator': {
                    filter: 'invert(1)', // Makes it white
                    cursor: 'pointer',
                  }
                }}
              />
              <TextField
                label="To Date"
                type="date"
                value={dateFilter.endDate}
                onChange={(e) => setDateFilter(prev => ({ ...prev, endDate: e.target.value }))}
                InputLabelProps={{ shrink: true }}
                sx={{
                  minWidth: 150,
                  '& input[type="date"]::-webkit-calendar-picker-indicator': {
                    filter: 'invert(1)', // Makes it white
                    cursor: 'pointer',
                  }
                }}
              />
              <Button
                variant="outlined"
                onClick={() => setDateFilter({ startDate: '', endDate: '' })}
                disabled={!hasActiveDateFilter}
              >
                Clear Dates
              </Button>
            </Box>
          </Box>

          {/* Groups */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, justifyContent: "center" }}>
            {Object.entries(availableTags).map(([group, values]) => (
              <Box
                key={group}
                sx={{
                  width: 260,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  p: 1.5,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography sx={{ textTransform: 'capitalize', fontWeight: 600 }}>{fieldPrettyNames[group]}</Typography>
                  <Button size="small" onClick={() => clearGroup(group)}>Clear</Button>
                </Box>

                <Box sx={{ maxHeight: 240, overflowY: 'auto', pr: 1 }}>
                  <FormGroup>
                    {values.map((tag) => (
                      <FormControlLabel
                        key={tag}
                        control={
                          <Checkbox
                            checked={selectedTags[group]?.includes(tag) || false}
                            onChange={() => handleToggle(group, tag)}  // live toggle
                          />
                        }
                        label={tag}
                      />
                    ))}
                  </FormGroup>
                </Box>
              </Box>
            ))}
          </Box>

          {/* Text Filter Fields */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, justifyContent: "center", marginTop: "35px" }}>
            {Object.entries(textFilterFields).map(([group, value]) => {
              return (
                <TextField
                  key={group}
                  label={group}
                  value={value}
                  onChange={(e) => {
                    const textFilterFieldsCopy = { ...textFilterFields }
                    textFilterFieldsCopy[group] = e.target.value
                    setTextFilterFields(textFilterFieldsCopy)
                  }}
                />
              );
            })}
          </Box>

        </DialogContent>

        {/* Optional tiny footer: just Close + Clear All */}
        <DialogActions>
          <Button onClick={clearAll}>Clear All</Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button onClick={closeFilters}>Close</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}