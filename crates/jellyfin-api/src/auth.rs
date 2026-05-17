/// Identity sent in every request via the `X-Emby-Authorization` header.
/// Jellyfin uses this for session bookkeeping (which device is connected,
/// what version, etc.). `device_id` should be persistent — a fresh UUID
/// per process would create a new "session" on the server each run.
#[derive(Debug, Clone)]
pub struct Identity {
    pub client: String,
    pub device: String,
    pub device_id: String,
    pub version: String,
}

impl Identity {
    pub fn new(
        client: impl Into<String>,
        device: impl Into<String>,
        device_id: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            client: client.into(),
            device: device.into(),
            device_id: device_id.into(),
            version: version.into(),
        }
    }
}

/// Build the `X-Emby-Authorization` header value. The token is omitted
/// before sign-in and included afterwards. Field names must be `Client`,
/// `Device`, `DeviceId`, `Version`, `Token` — case-sensitive on the server.
pub fn authorization_header(identity: &Identity, token: Option<&str>) -> String {
    let mut parts = vec![
        format!("Client=\"{}\"", escape_quotes(&identity.client)),
        format!("Device=\"{}\"", escape_quotes(&identity.device)),
        format!("DeviceId=\"{}\"", escape_quotes(&identity.device_id)),
        format!("Version=\"{}\"", escape_quotes(&identity.version)),
    ];
    if let Some(t) = token {
        parts.push(format!("Token=\"{}\"", escape_quotes(t)));
    }
    format!("MediaBrowser {}", parts.join(", "))
}

fn escape_quotes(s: &str) -> String {
    s.replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_without_token() {
        let id = Identity::new("Jelly", "Mac", "uuid-123", "0.1");
        let h = authorization_header(&id, None);
        assert_eq!(
            h,
            r#"MediaBrowser Client="Jelly", Device="Mac", DeviceId="uuid-123", Version="0.1""#
        );
    }

    #[test]
    fn header_with_token() {
        let id = Identity::new("Jelly", "Mac", "uuid-123", "0.1");
        let h = authorization_header(&id, Some("abc"));
        assert!(h.ends_with(r#"Token="abc""#));
    }

    #[test]
    fn escapes_embedded_quotes() {
        let id = Identity::new("Jelly", r#"Mac "Pro""#, "uuid", "0.1");
        let h = authorization_header(&id, None);
        assert!(h.contains(r#"Device="Mac \"Pro\"""#));
    }
}
