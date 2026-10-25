(function () {
  var article = document.getElementById("main-content");
  var sidebar = document.getElementById("reading-sidebar");
  var navigation = document.getElementById("reading-nav");
  var contentsToggle = document.getElementById("contents-toggle");
  var progress = document.getElementById("reading-progress");
  var progressFill = document.getElementById("reading-progress-fill");
  var focusToggle = document.getElementById("focus-toggle");
  var root = document.documentElement;
  var mobileLayout = window.matchMedia("(max-width: 1100px)");
  var textScale = 1;

  document.documentElement.classList.add("reading-enhanced");

  if (!article || !sidebar || !navigation) {
    return;
  }

  var tableHeading = article.querySelector("#table-of-contents");
  var tableList = tableHeading && tableHeading.nextElementSibling;

  if (tableHeading && tableList && tableList.tagName === "OL") {
    navigation.appendChild(tableHeading);
    navigation.appendChild(tableList);
    sidebar.classList.add("has-navigation");
    document.getElementById("reading-layout").classList.add("has-reading-navigation");
  }

  function closeContents() {
    sidebar.classList.remove("is-open");
    if (contentsToggle) {
      contentsToggle.setAttribute("aria-expanded", "false");
    }
  }

  if (contentsToggle) {
    contentsToggle.addEventListener("click", function () {
      var isOpen = sidebar.classList.toggle("is-open");
      contentsToggle.setAttribute("aria-expanded", String(isOpen));
    });
  }

  navigation.addEventListener("click", function (event) {
    if (event.target.closest("a")) {
      closeContents();
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeContents();
    }
  });

  mobileLayout.addEventListener("change", closeContents);

  if (focusToggle) {
    focusToggle.addEventListener("click", function () {
      var isFocused = document.body.classList.toggle("focus-mode");
      focusToggle.setAttribute("aria-pressed", String(isFocused));
      focusToggle.textContent = isFocused ? "Exit focus" : "Focus mode";
      if (contentsToggle) {
        contentsToggle.disabled = isFocused;
      }
      closeContents();
    });
  }

  function changeTextScale(amount) {
    textScale = Math.max(0.9, Math.min(1.25, textScale + amount));
    root.style.setProperty("--reading-scale", textScale.toFixed(2));
  }

  document.getElementById("text-smaller").addEventListener("click", function () {
    changeTextScale(-0.05);
  });

  document.getElementById("text-larger").addEventListener("click", function () {
    changeTextScale(0.05);
  });

  var navLinks = Array.prototype.slice.call(navigation.querySelectorAll('a[href^="#"]'));
  var observedHeadings = navLinks.map(function (link) {
    return document.getElementById(decodeURIComponent(link.getAttribute("href").slice(1)));
  }).filter(Boolean);

  if ("IntersectionObserver" in window && observedHeadings.length) {
    var headingObserver = new IntersectionObserver(function (entries) {
      var visible = entries.filter(function (entry) {
        return entry.isIntersecting;
      }).sort(function (first, second) {
        return first.boundingClientRect.top - second.boundingClientRect.top;
      });

      if (!visible.length) {
        return;
      }

      var activeId = visible[0].target.id;
      navLinks.forEach(function (link) {
        if (link.getAttribute("href") === "#" + activeId) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    }, { rootMargin: "-12% 0px -72% 0px" });

    observedHeadings.forEach(function (heading) {
      headingObserver.observe(heading);
    });
  }

  var scrollPending = false;

  function updateProgress() {
    var scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
    var percentage = scrollableHeight > 0
      ? Math.round((window.scrollY / scrollableHeight) * 100)
      : 0;

    progressFill.style.width = percentage + "%";
    progress.setAttribute("aria-valuenow", String(percentage));
    scrollPending = false;
  }

  window.addEventListener("scroll", function () {
    if (!scrollPending) {
      scrollPending = true;
      window.requestAnimationFrame(updateProgress);
    }
  }, { passive: true });

  window.addEventListener("resize", updateProgress);
  updateProgress();
}());
