// Minimal Express-side helper functions for the ReUse Campus ML service.
// Keep the ML service URL in an environment variable in production.
const ML_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

async function predictListingCategory({ title, description }) {
  const response = await fetch(`${ML_URL}/predict-category`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title || "", description: description || "" }),
  });
  if (!response.ok) throw new Error(`ML category service failed: ${response.status}`);
  return response.json();
}

async function recommendListings({ query = "", category, listingType, location, limit = 5, listings = [], excludeListingIds = [] }) {
  const response = await fetch(`${ML_URL}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      category: category || null,
      listing_type: listingType || null,
      location: location || null,
      limit,
      items: listings, // Must be current MongoDB listing objects with their real _id.
      exclude_listing_ids: excludeListingIds,
    }),
  });
  if (!response.ok) throw new Error(`ML recommendation service failed: ${response.status}`);
  return response.json();
}

module.exports = { predictListingCategory, recommendListings };
